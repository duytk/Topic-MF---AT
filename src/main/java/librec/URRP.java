package librec;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import librec.data.DataSplitter;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.util.Logs;
import static librec.util.Gamma.digamma;

import org.apache.commons.lang.StringUtils;
import org.json.JSONObject;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Table;
import com.uttesh.exude.stemming.Stemmer;



public class URRP {
	private static BiMap<String, Integer> userIds, itemIds , wordIds;
	private static int numFactor = 5;
	private static BiMap<List<String>, Integer> docIds;
	private static Set<Double> rati;
	private static SparseMatrix trainMatrix;
	private static SparseMatrix testMatrix;
	private static SparseMatrix validMatrix;
	private static DenseVector alpha,beta,lamda;
	static Table<Integer,Integer,Double> dataRatingTable;
	static Table<Integer,Integer,Integer> dataReviewTable;
	static Table<Integer,Integer,Integer> x;
	static Table<Integer,Integer,Integer> z;
	static List<Double> ratingScale;
	static DenseMatrix Puk;
	static DenseMatrix Pkw;
	static double[][][] Pkir;
	static DenseMatrix Nkw;
	static DenseVector Nk;
	
	static DenseMatrix Nuk;
	static DenseVector Nu;
	
	static DenseMatrix Muk;
	static DenseVector Mu;
	
	static int[][][] Ckvs;
	static DenseMatrix Ckv;
	// size of statistics
	static int numStats;
	
	//posterior probabilities
	static DenseMatrix pPuk;
	static DenseMatrix pPkw;
	static double[][][] pPkir;
	private static double preMSE;
	public static String[] readStopWords(String file){
		String[] stopWords = null;
		
		try{
			Scanner sfile = new Scanner(new File(file));
			int n = sfile.nextInt();
			stopWords = new String[n];
			for(int i =0; i < n ; i++){
				stopWords[i] = sfile.next();	
			}
			sfile.close();
		}catch(IOException e){
			e.printStackTrace();
		}
		return stopWords;
	}
	
	public static Boolean isStopWord(String word, String[] stopWords){
		boolean found = false;
		int min = 0, max = stopWords.length - 1,  // specifies the range
			    mid,                 // midpoint
			    result;              // result of comparing words
			
			while (!found && (min <= max)) 
			    {
				mid = (min + max) / 2;
				result = compareWords(word, stopWords[mid]);
				if (result == 0)  // found it
				    found = true;
				else if (result < 0) // in the first half
				    max = mid - 1;
				else // in the second half
				    min = mid + 1; 
			    }

			return found;
	}
	 public static int compareWords(String word1, String word2)
	    {
		return word1.compareToIgnoreCase(word2);
	    }
	 
	// khoi tao 
	protected static void init(){
		userIds = HashBiMap.create();
		itemIds = HashBiMap.create();
		wordIds = HashBiMap.create();
		docIds = HashBiMap.create();
		rati = new HashSet<>();
		dataRatingTable = HashBasedTable.create();
		dataReviewTable = HashBasedTable.create();
		//attitude-assignments for each rating
		x = HashBasedTable.create();
		// topic-assignments for each word
		z = HashBasedTable.create();
		Stemmer s = new Stemmer();
		String[] stopWords = readStopWords("sw.txt");
		BufferedReader br = null;
		try{
			String line;
			br = new BufferedReader(new FileReader("D:\\a.json"));
			while ((line = br.readLine()) != null){
				JSONObject json = new JSONObject(line);
				String user = json.getString("reviewerID");
				int row = userIds.containsKey(user) ? userIds.get(user) : userIds.size();
				userIds.put(user, row);
				String item = json.getString("asin");
				int col = itemIds.containsKey(item) ? itemIds.get(item) : itemIds.size();
				Double rate = json.getDouble("overall");
				rati.add(rate);
				itemIds.put(item, col);
				dataRatingTable.put(row, col, rate);
				String reviewText = json.getString("reviewText");
				reviewText = reviewText.toLowerCase();
				reviewText= reviewText.trim().replace("\\s+", "").replace(".", "").replace(",", "").replace("-", "").replace("!", "").replace("(", "").replace(")", "")
							.replace("&quot;", "").replace("%", "").replace("#", "").replace("*", "").replace("?", "").replace("'", "").replace(":", "").replace("$", "")
							.replace(";", "").replace("&amp;", "");
				String[] words = reviewText.split(" ");
				List<String> wordList = new ArrayList<>();
				for(int m=0; m < words.length ; m++){
					if(isStopWord(words[m],stopWords)== false && StringUtils.isNumeric(words[m]) == false){
						words[m] = s.stem(words[m]);
						wordList.add(words[m]);
						int colW = wordIds.containsKey(words[m]) ? wordIds.get(words[m]) : wordIds.size();
						wordIds.put(words[m], colW);
					}
				}
				int rowD = docIds.containsKey(wordList) ? docIds.get(wordList) : docIds.size();
				dataReviewTable.put(row, col, rowD);
				docIds.put(wordList, rowD);
			}
		}catch(IOException e){
			e.printStackTrace();
		}finally{
			try{
				if( br !=null) br.close();
			}catch(IOException ex){
				ex.printStackTrace();
			}
		}
		ratingScale = new ArrayList<>(rati);
		Collections.sort(ratingScale);
		SparseMatrix sm1 = new SparseMatrix(userIds.size(),itemIds.size(),dataRatingTable);
		DataSplitter ds = new DataSplitter(sm1);
		SparseMatrix[] data = ds.getRatio(0.8,0.1);
		trainMatrix = data[0];
		validMatrix = data[1];
		testMatrix = data[2];
		
		alpha = new DenseVector(numFactor);
		alpha.setAll(0.1);
		beta = new DenseVector(wordIds.size());
		beta.setAll(0.1);
		lamda = new DenseVector(ratingScale.size());
		lamda.setAll(0.1);
		
		//phi
		Puk = new DenseMatrix(userIds.size(),numFactor);
		//theta
		Pkw = new DenseMatrix(numFactor,wordIds.size());
		// poison
		Pkir = new double[numFactor][itemIds.size()][ratingScale.size()];
		
		
		// bien dem
		Nkw = new DenseMatrix(numFactor,wordIds.size());
		Nk = new DenseVector(numFactor);
		
		Nuk = new DenseMatrix(userIds.size(), numFactor);
		Nu = new DenseVector(userIds.size());
		
		Muk = new DenseMatrix(userIds.size(),numFactor);
		Mu = new DenseVector(userIds.size());
		
		Ckvs = new int[numFactor][itemIds.size()][ratingScale.size()];
		Ckv = new DenseMatrix(numFactor,itemIds.size());
		// khoi tao cac bien dem
		
		for(MatrixEntry me : trainMatrix){
			int u = me.row();
			int v = me.column();
			double ruv = me.get();
			int r = ratingScale.indexOf(ruv);
			int t1 = (int) (Math.random()*numFactor);
			
			x.put(u, v, t1);
			Muk.add(u,t1,1);
			Mu.add(u,1);
			Ckvs[t1][v][r] ++;
			Ckv.add(t1,v,1);
			List<String> ls = docIds.inverse().get(dataReviewTable.get(u, v));
			int m =docIds.get(ls);
			for(String w : ls){
				int t2 = (int) (Math.random() * numFactor);
				int n = wordIds.get(w);
				z.put(m, n, t2);
				Nkw.add(t2,n,1);
				Nk.add(t2,1);
				Nuk.add(u, t2, 1);
				Nu.add(u,1);
			}
		}
	}
	
	//cap nhat chu de va attitude (5),(6)
	protected static void eStep(){
		double sumAlpha = alpha.sum();
		double sumBeta = beta.sum();
		double sumLamda = lamda.sum();
		// collapse Gibbs
		
		for(MatrixEntry me : trainMatrix){
			int u = me.row();
			int v = me.column();
			double ruv = me.get();
			int r = ratingScale.indexOf(ruv);
			int t1 = x.get(u, v);
			Muk.add(u,t1,-1);
			Mu.add(u,-1);
			Ckvs[t1][v][r]--;
			Ckv.add(t1, v, -1);
			
			//conditional probabilities for Xj
			double[] p1 = new double[numFactor];
			for(int k=0; k< numFactor; k++){
				p1[k] = (Nuk.get(u, k) + Muk.get(u, k) + alpha.get(k))/
						(Nu.get(u) + Mu.get(u) + sumAlpha )*
						(Ckvs[k][v][r] + lamda.get(r))/
						(Ckv.get(k, v) + sumLamda);
			}
			// cumulate multinomial parameters
			for(int k =1; k< p1.length; k++){
				p1[k] += p1[k-1];
			}
			// scaled sample because of unnormalized p[],
			double rand1 = Math.random() * p1[numFactor - 1];
			for(t1 =0; t1<p1.length; t1++){
				if(rand1 <p1[t1]){
					break;
				}
			}
			x.put(u, v, t1);
			Muk.add(u,t1,1);
			Mu.add(u,1);
			Ckvs[t1][v][r]++;
			Ckv.add(t1, v,1);
			
			//conditional probabilities for Zi
			List<String> ls = docIds.inverse().get(dataReviewTable.get(u, v));
			int m =docIds.get(ls);
			for(String w : ls){
				int n = wordIds.get(w);
				int t2 = z.get(m, n);
				Nkw.add(t2, n, -1);
				Nk.add(t2, -1);
				Nuk.add(u,t2,-1);
				Nu.add(u,-1);
				double[] p2 = new double[numFactor];
				for(int k = 0;k < numFactor ; k++){
					p2[k] = (Nuk.get(u, k) + Muk.get(u, k) + alpha.get(k))/
							(Nu.get(u) + Mu.get(u) + sumAlpha)*
							(Nkw.get(k, n) + beta.get(n))/
							(Nk.get(k) + sumBeta);
				}
				for (int k = 1; k < p2.length; k++) {
					p2[k] += p2[k - 1];
				}
				double rand2 = Math.random() * p2[numFactor - 1];
				for (t2 = 0; t2 < p2.length; t2++) {
					if (rand2 < p2[t2])
						break;
				}
				// new topic
				z.put(m, n, t2);
				Nkw.add(t2, n, 1);
				Nk.add(t2, 1);
				Nuk.add(u,t2,1);
				Nu.add(u,1);
			}
		}
	}
	
	// cap nhat hyperparameter ( theo bai bao cap nhat Estimating a Dirichlet distribution Thomas P. Minka)
	protected static void mStep(){
		double sumAlpha = alpha.sum();
		double sumBeta = beta.sum();
		double sumLamda = lamda.sum();
		double ak , bw , lr;
		
		// update alpha vector
		for(int k =0; k< numFactor; k++){
			ak = alpha.get(k);
			double numerator = 0, denominator =0;
			for(int u=0 ; u < userIds.size(); u++){
				numerator += digamma(Nuk.get(u, k) + Muk.get(u, k)+ ak) - digamma(ak);
				denominator += digamma(Nu.get(u) + Mu.get(u) + sumAlpha) - digamma(sumAlpha);
			}
			if(denominator !=0){
				alpha.set(k, ak*(numerator/denominator));
			}
		}
		
		// update lamda vector
		for(int sr =0; sr< ratingScale.size(); sr++){
			lr = lamda.get(sr);
			double numerator =0 , denominator =0;
			for( int v = 0; v< itemIds.size(); v++){
				for (int k =0 ; k< numFactor; k++){
					numerator += digamma(Ckvs[k][v][sr] + lr) -digamma(lr);
					denominator += digamma(Ckv.get(k, v) + sumLamda) - digamma(sumLamda);
				}
			}
			if( numerator !=0){
				lamda.set(sr, lr * (numerator/denominator));
			}
		}
		
		// update beta vector
		for(int w = 0; w< wordIds.size(); w++){
			bw = beta.get(w);
			double numerator =0 , denominator =0;
			for(int k =0; k < numFactor; k++){
				numerator += digamma(Nkw.get(k, w) + bw) - digamma(bw);
				denominator += digamma(Nk.get(k) + sumBeta) -digamma(sumBeta);
			}
			if( numerator !=0){
				beta.set(w, bw *(numerator/denominator));
			}
		}
	}
	
	//(7),(8),(9)
	protected static void readoutParams(){
		double sumAlpha = alpha.sum();
		double sumBeta = beta.sum();
		double sumLamda = lamda.sum();
		double val =0;
		
		// phi - puk
		for(int u= 0; u < userIds.size(); u++){
			for(int k =0; k< numFactor; k++){
				val = (Nuk.get(u, k) + Muk.get(u, k) + alpha.get(k))/ (Nu.get(u) + Mu.get(u) + sumAlpha);
				Puk.add(u, k, val);
			}
		}
		
		// theta - pkw
		
		for(int k=0; k<numFactor; k++){
			for(int w = 0; w< wordIds.size(); w++){
				val = (Nkw.get(k, w) + beta.get(w))/(Nk.get(k) + sumBeta);
			}
		}
		
		//  poison - pkir
		for(int k =0 ; k< numFactor; k++){
			for(int v =0; v< itemIds.size(); v++){
				for( int s=0 ; s< ratingScale.size(); s++){
					val = (Ckvs[k][v][s] + lamda.get(s))/ (Ckv.get(k, v) + sumLamda);
					Pkir[k][v][s] += val;
				}
			}
		}
		numStats++;
	}
	
	protected static void estimateParams(){
		pPuk = Puk.scale(1.0/numStats);
		pPkw = Pkw.scale(1.0/numStats);
		
		pPkir = new double[numFactor][itemIds.size()][ratingScale.size()];
		for(int k =0 ; k< numFactor; k++){
			for(int v =0 ; v<itemIds.size(); v++){
				for(int s = 0; s < ratingScale.size(); s++){
					pPkir[k][v][s] = Pkir[k][v][s]/numStats;
				}
			}
		}
	}
	
	//prediction score(11)
	protected static double predict(int u , int v){
		double pred = 0;
		for(int s =0; s< ratingScale.size(); s++){
			double rate = ratingScale.get(s);
			double prob = 0;
			for(int k =0 ; k< numFactor; k++){
				prob += pPuk.get(u, k) *pPkir[k][v][s];			
			}
			pred += prob*rate;
		}
		return pred;
	}
	
	protected static boolean isConverged(int iter){
		if(validMatrix == null){
			return false;
		}
		estimateParams();
		int numCount =0;
		double sum =0;
		for(MatrixEntry me : validMatrix){
			int u = me.row();
			int v = me.column();
			double rate = me.get();
			double pred = predict(u,v);
			if(pred > ratingScale.get(ratingScale.size()-1)){
				pred = ratingScale.get(ratingScale.size()-1);
			}
			if(pred < ratingScale.get(0) ){
				pred = ratingScale.get(0);
			}
			if(Double.isNaN(pred)){
				continue;
			}
			double err = rate - pred;
			sum += err*err;
			numCount++;
		}
		double MSE = sum/numCount;
		double delta = MSE - preMSE;
		Logs.debug(" iter {} achieves MSE = {}, delta_MSE = {}",iter, (float) MSE,
				(float) (delta));
		if(numStats >1 && delta>0){
			return true;
		}
		preMSE = MSE;
		return false;
		
	}
	public static void main(String[] args) throws Exception{
		// khoi tao data
		init();
		int iter = 50;
		int burn_in = 30;
		int sampleLag = 10;
		for(int i = 1 ; i <= iter; i++){
			
			
			//eStep : cap nhat lai tocpic va attitude
			eStep();
			
			// mStep cap nhat lai cac tham so alpha, beta , lamda
			mStep();
			if((i > burn_in) && (i% sampleLag == 0)){
				readoutParams();
				if(isConverged(i)){
					break;
				}
			}
		}
		estimateParams();
		int numCount =0;
		double sum =0;
		for( MatrixEntry me : testMatrix){
			double rate = me.get();
			int u = me.row();
			int v = me.column();
			double pre = predict(u,v);
			if(pre > ratingScale.get(ratingScale.size()-1)){
				pre = ratingScale.get(ratingScale.size()-1);
			}
			if(pre < ratingScale.get(0) ){
				pre = ratingScale.get(0);
			}
			if(Double.isNaN(pre)){
				continue;
			}
			double err = rate - pre;
			sum += err*err;
			numCount++;
		}
		System.out.println("MSE :" + sum/numCount );
		
	}
}