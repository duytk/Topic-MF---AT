package librec;

public class Review{
	private int userId;
	private int itemId;
	private int reviewId;
	private String[] word;
	private int predictValue;
	//
	 
	public int getUserId() {
		return userId;
	}
	public void setUserId(int userId) {
		this.userId = userId;
	}
	public int getItemId() {
		return itemId;
	}
	public void setItemId(int itemId) {
		this.itemId = itemId;
	}
	public int getReviewId() {
		return reviewId;
	}
	public void setReviewId(int reviewId) {
		this.reviewId = reviewId;
	}
	public String[] getWord() {
		return word;
	}
	public void setWord(String[] word) {
		this.word = word;
	}
	public int getPredictValue() {
		return predictValue;
	}
	public void setPredictValue(int predictValue) {
		this.predictValue = predictValue;
	}
}