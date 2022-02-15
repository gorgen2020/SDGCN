package myutils;

//for sorting a list
public class RankedObject implements Comparable<RankedObject> {
	public Object name;
	public double score;

	public RankedObject(Object object, double score) {
		this.name = object;
		this.score = score;
	}

	@Override
	public int compareTo(RankedObject other) {
		if (this.score < other.score)
			return 1;
		if (this.score > other.score)
			return -1;
		return 0;
	}

	@Override
	public String toString() {
		return name.toString()+"\t"+score;
	}
}