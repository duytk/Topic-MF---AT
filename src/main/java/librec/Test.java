package librec;

import static com.github.lbfgs4j.liblbfgs.LbfgsConstant.LineSearch.LBFGS_LINESEARCH_DEFAULT;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import librec.data.DenseMatrix;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;

import org.apache.commons.lang.StringUtils;
import org.json.JSONObject;

import com.github.lbfgs4j.LbfgsMinimizer;
import com.github.lbfgs4j.liblbfgs.Function;
import com.github.lbfgs4j.liblbfgs.LbfgsConstant.LBFGS_Param;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.HashBiMap;
import com.google.common.collect.Table;
import com.uttesh.exude.stemming.Stemmer;

public class Test{
	private static BiMap<String, Integer> userIds, itemIds , wordIds;
	private static BiMap<Integer,Integer> reviewIds;
	private static BiMap<List<String>, Integer> docIds;
	private static int numFactor = 5;
	private static double lamdaU,lamdaV,lamdaB = 0.001;
	public static void main(String[] args) throws Exception{
		Table<Integer,Integer,Double> dataRatingTable = HashBasedTable.create();
		final Table<Integer,Integer,Float> dataWordToReviewTable = HashBasedTable.create();
		final Table<Integer,Integer,Integer> dataReviewTable = HashBasedTable.create();
		reviewIds = HashBiMap.create();
		userIds = HashBiMap.create();
		itemIds = HashBiMap.create();
		wordIds = HashBiMap.create();
		docIds = HashBiMap.create();
		final HashMap<Integer, List<Integer>> hm1 = new HashMap<Integer, List<Integer>>();
		final HashMap<Integer, List<Integer>> hm2 = new HashMap<Integer, List<Integer>>();
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
		
		for(int i =0 ; i<wordIds.size(); i++){
			String word = wordIds.inverse().get(i);
			for(int j =0 ; j<docIds.size(); j++){
				List<String> wordList = docIds.inverse().get(j);
				if(Collections.frequency(wordList, word)>0){
					float f = (float)Collections.frequency(wordList, word)/wordList.size();
					dataWordToReviewTable.put(j, i, f);
				}
			}
		}
		final SparseMatrix sm1 = new SparseMatrix(userIds.size(),itemIds.size(),dataRatingTable);
		// x[0] là muy
		// x[1] là kappa
		// x[2 -> (K+1)] là beta user
		// x[(K+2) -> (2K +1)] là beta item
		// x[2K+2 -> nUser * K + 2K + 1] là gamma User
		// x[nUser*K + 2K +2 -> nUser*K + nItem*K + 2K+1] là gamma Item
		int m =3;
		int n = userIds.size() + 3;
		int p = userIds.size() + itemIds.size() + 3;
		int q = userIds.size() + itemIds.size() + numFactor*userIds.size()+3;	
		for(MatrixEntry me :sm1){
			int u = me.row();
			int v = me.column();
//			System.out.println(dataReviewTable.get(u, v));
			if(hm1.get(u) == null){
				List<Integer> ls = new ArrayList<>();
				ls.add(m);
				ls.add(p);
				hm1.put(u, ls);
				m++;
				p = p +5;
			}
			if(hm2.get(v) == null){
				List<Integer> ls = new ArrayList<>();
				ls.add(n);
				ls.add(q);
				hm2.put(v, ls);
				n++;
				q = q +5;
			}
			
		}
//		SparseMatrix sm2 = new SparseMatrix(userIds.size(),itemIds.size(),dataReviewTable);
//		SparseMatrix sm3 = new SparseMatrix(docIds.size(),wordIds.size(),dataWordToReviewTable);
//		DataSplitter ds = new DataSplitter(sm);
//		SparseMatrix[] n = ds.getRatioByItem(0.8);
//		SparseMatrix train = n[0];
//		SparseMatrix test = n[1];
//		DenseVector userBias = new DenseVector(userIds.size());
//		DenseMatrix P = new DenseMatrix(userIds.size(),numFactor);
//		DenseMatrix Q = new DenseMatrix(itemIds.size(),numFactor);
//		DenseVector itemBias = new DenseVector(itemIds.size());
//		DenseMatrix Theta = new DenseMatrix(docIds.size(),numFactor);
		final DenseMatrix Omega = new DenseMatrix(wordIds.size(),numFactor);
		Omega.init(0.01);
		
			
//			List<String> ls = docIds.inverse().get(dataReviewTable.get(0, 0));
//			Set<String> set = new HashSet<String>(ls);
//			for(String w : set){
//				float f = dataWordToReviewTable.get(dataReviewTable.get(0, 0),wordIds.get(w));
//				System.out.println("word " + w + " fre = " + f);
//				
//			}
//			

		Function f = new Function(){

			@Override
			public int getDimension() {				
				return 1+2+(numFactor +1)*(userIds.size() + itemIds.size())
//						+ numFactor*wordIds.size()
						;
			}
			
			@Override
			public double valueAt(double[] x) {
				double res = 0;
				// tính Lrating:
				for(MatrixEntry me : sm1){
					double thetadij = 0;
					double ruj = me.get();
					int u = me.row();
					int v = me.column();
					List<Integer> lu = hm1.get(u);
					List<Integer> li = hm2.get(v);
					int a = lu.get(0);
					int b = lu.get(1);
					int c = li.get(0);
					int d = li.get(1);
					double euj = x[0] + x[a] + x[c] - ruj + x[b]*x[d] + x[b+1]*x[d+1] + x[b+2]*x[d+2] + x[b+3]*x[d+3] + x[b+4]*x[d+4];
					// binh phuong euj
					res += Math.pow(euj, 2);
					
					// thetadij
					for(int i = 0; i<5; i++){
						thetadij += Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5));
					}
					List<String> ls = docIds.inverse().get(dataReviewTable.get(u, v));
					Set<String> set = new HashSet<String>(ls);
					for(String w : set){
						float f = dataWordToReviewTable.get(dataReviewTable.get(u, v),wordIds.get(w));
						double mul =0;
						for(int i =0; i<5; i++){
							mul+= (Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij) * Omega.row(wordIds.get(w), true).get(i);
						}
						res += Math.pow(mul -f,2) ;
					}
				}
				for(int i = 3 ; i<= userIds.size() + 2 ;i++){
					res += lamdaB*x[i]*x[i];
				}
				for(int i = userIds.size() + 3; i<= userIds.size() + itemIds.size() + 2 ; i++){
					res += lamdaB*x[i] * x[i];
				}
				for(int i=  userIds.size() + itemIds.size() + 3; i<= userIds.size() + itemIds.size() + numFactor*userIds.size()+2;i++ ){
					res += lamdaU*x[i]*x[i];
				}
				for(int i = userIds.size() + itemIds.size() + numFactor*userIds.size()+3; i<=1 +(numFactor +1)*(userIds.size() + itemIds.size());i++){
					res += lamdaV*x[i]*x[i];
				}
				return res;
			}

			@Override
			public double[] gradientAt(double[] x) {
				double[] g = new double[1+2 +(numFactor +1)*(userIds.size() + itemIds.size())];
				for(MatrixEntry me : sm1){
					double thetadij = 0;
					double ruj = me.get();
					int u = me.row();
					int v = me.column();
					List<Integer> lu = hm1.get(u);
					List<Integer> li = hm2.get(v);
					List<String> ls = docIds.inverse().get(dataReviewTable.get(u, v));
					Set<String> set = new HashSet<String>(ls);
					int a = lu.get(0);
					int b = lu.get(1);
					int c = li.get(0);
					int d = li.get(1);
					
					// tính đạo hàm
					g[0] += 2*(x[0]+ x[a] + x[c] - ruj + x[b]*x[d] + x[b+1]*x[d+1] + x[b+2]*x[d+2] + x[b+3]*x[d+3] + x[b+4]*x[d+4]);
					g[a] += 2*(x[0]+ x[a] + x[c] - ruj + x[b]*x[d] + x[b+1]*x[d+1] + x[b+2]*x[d+2] + x[b+3]*x[d+3] + x[b+4]*x[d+4]);
					g[c] += 2*(x[0]+ x[a] + x[c] - ruj + x[b]*x[d] + x[b+1]*x[d+1] + x[b+2]*x[d+2] + x[b+3]*x[d+3] + x[b+4]*x[d+4]);
					
					
					// dao ham tung bien u va v
					for(int i = 0; i<5; i++){
						thetadij += Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5));
						g[b+i] += 2*(x[d+i]*(x[0]+ x[a] + x[c] - ruj + x[b]*x[d] + x[b+1]*x[d+1] + x[b+2]*x[d+2] + x[b+3]*x[d+3] + x[b+4]*x[d+4]));
						g[d+i] += 2*(x[b+i]*(x[0]+ x[a] + x[c] - ruj + x[b]*x[d] + x[b+1]*x[d+1] + x[b+2]*x[d+2] + x[b+3]*x[d+3] + x[b+4]*x[d+4]));
					}
					
					
					
					for(String w : set){
						Float f = dataWordToReviewTable.get(dataReviewTable.get(u, v),wordIds.get(w));
						double mul =0;
						//mul la (thetadij)(omegan)^T
							for(int i =0; i<5; i++){
								mul+= (Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij) * Omega.row(wordIds.get(w), true).get(i);
							}
						
							// dao ham cua K
							for(int i = 0;i<5;i++){
								g[1] += 2*(mul-f)*(Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij) * Omega.row(wordIds.get(w), true).get(i)
										*(1- (Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij))*Math.pow(x[b+i]*x[b+i],0.5);
								g[2] += 2*(mul-f)*(Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij) * Omega.row(wordIds.get(w), true).get(i)
										*(1- (Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij))*Math.pow(x[d+i]*x[d+i],0.5);
							}
//							 dao ham u v theo review
							for(int i =0;i<5;i++){
								double t1 = x[b+i]/Math.abs(x[b+i]);
								double t2 = x[d+i]/Math.abs(x[d+i]);
 							g[b+i] += 2*x[1]*(mul-f)*(Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij)
									  *(1-(Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij))	
									  *Omega.row(wordIds.get(w), true).get(i);
							g[d+i] += 2*x[2]*(mul-f)*(Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij)
									  *(1-(Math.exp(x[1]*Math.pow(x[b+i]*x[b+i],0.5) + x[2]*Math.pow(x[d+i]*x[d+i],0.5))/thetadij))	
									  *Omega.row(wordIds.get(w), true).get(i);
							}
					}
				}
					
				for(int i = 3 ; i<= userIds.size() + 2 ;i++){
					g[i] += 2*lamdaB*x[i];
				}
				for(int i = userIds.size() + 3; i<= userIds.size() + itemIds.size() + 2 ; i++){
					g[i] += 2*lamdaB*x[i];
				}
				for(int i=  userIds.size() + itemIds.size() + 3; i<= userIds.size() + itemIds.size() + numFactor*userIds.size()+2;i++ ){
					g[i] += 2*lamdaU*x[i];
				}
				for(int i = userIds.size() + itemIds.size() + numFactor*userIds.size()+3; i<=1 +(numFactor +1)*(userIds.size() + itemIds.size());i++){
					g[i] += 2*lamdaV*x[i];
				}
				return g; 
			}
			
		};
		

		boolean verbose = true;
		LBFGS_Param param = new LBFGS_Param(
				6, 1e-2, 0, 1e-2,
			    50, LBFGS_LINESEARCH_DEFAULT, 20,
			    1e-20, 1e20, 1e-4, 0.9, 0.9, 1.0e-16,
			    0.0, 0, -1
	    );
	    LbfgsMinimizer minimizer = new LbfgsMinimizer(verbose);
	    double[] x = minimizer.minimize(f);
	    double min = f.valueAt(x);

	    System.out.printf("The function achieves its minimum value = %.5f at: ", min);
	    printOut(x);
		}
	
//	public static double topic(int K ){
//		double res=0; 
//		return res;
//	}
	
	public static void printOut(double[] x) {
	    System.out.printf("[");
	    for (double v: x)
	      System.out.printf(" %f", v);
	    System.out.printf(" ]\n");	  
	  }
	
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
		
	
}