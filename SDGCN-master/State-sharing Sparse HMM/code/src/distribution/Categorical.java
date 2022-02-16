package distribution;

import java.util.*;

//for sampling words according to importance scores (e.g., idf scores and word similarity scores)
public class Categorical {
	ArrayList<Object> objects = new ArrayList<Object>();
	ArrayList<Double> cdf = new ArrayList<Double>(); // cdf stands for Cumulative Distribution Function
	Random random = new Random(1);

	public Categorical(HashMap object2score) {
		ArrayList<Double> scores = new ArrayList<Double>();
		for (Object object : object2score.keySet()) {
			Double score = (Double) object2score.get(object);
			objects.add(object);
			scores.add(score);
		}
		ArrayList<Double> pmf = normalize(scores);
		this.cdf = pmf2cdf(pmf);
	}

	public Object sample() {
		double randDouble = random.nextDouble();
		int start = 0, end = cdf.size() - 1;
		while (start != end) {
			int mid = (start + end) / 2;
			if (cdf.get(mid) > randDouble) {
				end = mid;
			} else {
				start = mid + 1;
			}
		}
		return objects.get(start);
	}

	private ArrayList<Double> normalize(ArrayList<Double> list) {
		ArrayList<Double> normalizedList = new ArrayList<Double>(); // pmf stands for Probability Mass Function
		double sum = 0;
		for (double num : list) {
			sum += num;
		}
		for (double num : list) {
			normalizedList.add(num / sum);
		}
		return normalizedList;
	}

	private ArrayList<Double> pmf2cdf(ArrayList<Double> pmf) {
		ArrayList<Double> cdf = new ArrayList<Double>();
		double sum = 0;
		for (double prob : pmf) {
			sum += prob;
			cdf.add(sum);
		}
		return cdf;
	}
}
