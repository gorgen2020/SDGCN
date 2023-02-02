package distribution;

import com.mongodb.BasicDBList;
import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import data.WordDataset;
import myutils.ArrayUtils;

import java.io.Serializable;
import java.util.*;

/**
 * Created by chao on 4/14/15.
 */
public class Multinomial implements Serializable {

	double[] prob = null;

	public Multinomial() {
	}

	public Multinomial(double[] prob) {
		this.prob = prob;
		ArrayUtils.normalize(this.prob);
	}

	public Multinomial(DBObject o) {
		load(o);
	}

	public void fit(int dimension, List<Integer> data, List<Double> weights) {
		if (data.size() != weights.size()) {
			System.out.println("Error when fitting the multinomial. Database and weight sizes are not equal!");
			return;
		}
		init(dimension);
		learnProb(data, weights);
	}

	public double[] getProb() {
		return prob;
	}

	private void init(int dimension) {
		prob = new double[dimension];
		Arrays.fill(prob, 0);
	}

	private void learnProb(List<Integer> data, List<Double> weights) {//???
		for (int i = 0; i < data.size(); i++) {
			double weight = weights.get(i);
			int index = data.get(i);
			prob[index] += weight;
		}

//		// Smoothing
//		double min = 1.0e-8 / prob.length;
//		//        for (int i=0; i < prob.length; i++) {
//		//            if (prob[i] != 0 && prob[i] < min)
//		//                min = prob[i];
//		//        }
//		for (int i = 0; i < prob.length; i++) {
//			prob[i] += min * 0.01;
//		}

		// Normalize to generate probability distribution.
		ArrayUtils.normalize(prob);
	}

	// Calc the log probability of generating the sample from this multinomial, the factorial constant is ignored.
	public double calcLL(int sample) {
		return calcLL(sample, false);
	}

	public double calcLL(int sample, boolean isTest) {
		return prob[sample];
//		double result = 0.0;
//		int totalCount = 0;
//		Iterator iter = sample.entrySet().iterator();
//		while (iter.hasNext()) {
//			Map.Entry<Integer, Integer> entry = (Map.Entry<Integer, Integer>) iter.next();
//			int index = entry.getKey(); // The position of the observed item.
//			int count = entry.getValue(); // The number of times that the item has appeared.
//			result += count * Math.log(prob[index]);
//			totalCount += count;
//		}
//		if (isTest) {
//			return result / totalCount;
//		} else {
//
//			return result;
//		}
	}

	@Override
	public String toString() {
		String result = "";
		for (int i = 0; i < prob.length; i++) {
//			if (prob[i] >= 1.0 / prob.length)
//				result += "(" + i + ":" + prob[i] + ") ";
			result += prob[i] + " ";
		}
		result += "\n";
		return result;
	}

	// Get the top-k words
	public String getWordDistribution(WordDataset wd, int K) {
		List<Pair> wordScorePairs = new ArrayList<Pair>();
		for (int i = 0; i < prob.length; i++) {
			wordScorePairs.add(new Pair(i, prob[i]));
		}
		Collections.sort(wordScorePairs, new Comparator<Pair>() {
			public int compare(Pair u1, Pair u2) {
				if (u1.getScore() - u2.getScore() > 0)
					return -1;
				else if (u1.getScore() - u2.getScore() == 0)
					return 0;
				else
					return 1;
			}
		});
		String result = "";
		for (int i = 0; i < K; i++) {
			Pair p = wordScorePairs.get(i);
			result += "(" + wd.getWord(p.getIndex()) + ":" + p.getScore() + ") ";
		}
		return result;
	}

	class Pair {
		int index;
		double score;

		public Pair(int index, double score) {
			this.index = index;
			this.score = score;
		}

		public int getIndex() {
			return index;
		}

		public double getScore() {
			return score;
		}
	}

//	public static void main(String[] args) {
//		Map<Integer, Integer> v1 = new HashMap<Integer, Integer>();
//		v1.put(0, 1);
//		v1.put(2, 1);
//		Map<Integer, Integer> v2 = new HashMap<Integer, Integer>();
//		v2.put(1, 5);
//		List<Map<Integer, Integer>> data = new ArrayList<Map<Integer, Integer>>();
//		data.add(v1);
//		data.add(v2);
//		System.out.println(data);
//		Double[] w = new Double[] { 1.0, 1.0 };
//		List weights = Arrays.asList(w);
//		System.out.println(weights);
//		Multinomial m = new Multinomial();
//		m.fit(4, data, weights);
//		double[] prob = m.getProb();
//		for (int i = 0; i < prob.length; i++) {
//			System.out.println(prob[i]);
//		}
//		System.out.println(m.calcLL(v1));
//		System.out.println(m);
//	}

	public DBObject toBSon() {
		return new BasicDBObject("prob", prob);
	}

	public void load(DBObject o) {
		Object[] pvalues = ((BasicDBList) o.get("prob")).toArray();
		this.prob = new double[pvalues.length];
		for (int i = 0; i < pvalues.length; i++)
			this.prob[i] = (Double) pvalues[i];
	}

}
