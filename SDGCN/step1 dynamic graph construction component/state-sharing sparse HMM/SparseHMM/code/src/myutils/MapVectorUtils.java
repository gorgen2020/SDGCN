package myutils;

import java.util.*;

//here a vector is represented by a HashMap
public class MapVectorUtils<Key> {
	public MapVectorUtils() {

	}

	public HashMap<Key, Double> scale(HashMap<Key, Double> vector, double scalar) {
		HashMap<Key, Double> newVec = new HashMap<Key, Double>();
		for (Key key : vector.keySet()) {
			newVec.put(key, vector.get(key) * scalar);
		}
		return newVec;
	}

	public HashMap<Key, Double> add(HashMap<Key, Double> vector1, HashMap<Key, Double> vector2) {
		HashMap<Key, Double> bag = new HashMap<Key, Double>();
		HashSet<Key> keySet = new HashSet<Key>();
		keySet.addAll(vector1.keySet());
		keySet.addAll(vector2.keySet());
		for (Key key : keySet) {
			Double value = (double) 0;
			if (vector1.containsKey(key)) {
				value += vector1.get(key);
			}
			if (vector2.containsKey(key)) {
				value += vector2.get(key);
			}
			bag.put(key, value);
		}
		return bag;
	}

	public HashMap<Key, Double> multiply(HashMap<Key, Double> vector1, HashMap<Key, Double> vector2) {
		HashMap<Key, Double> bag = new HashMap<Key, Double>();
		HashSet<Key> keySet = new HashSet<Key>();
		keySet.addAll(vector1.keySet());
		keySet.addAll(vector2.keySet());
		for (Key key : keySet) {
			Double value = (double) 1;
			if (vector1.containsKey(key)) {
				value *= vector1.get(key);
			} else {
				value = (double) 0;
			}
			if (vector2.containsKey(key)) {
				value *= vector2.get(key);
			} else {
				value = (double) 0;
			}
			bag.put(key, value);
		}
		return bag;
	}

	public double cosine(HashMap<Key, Double> vector1, HashMap<Key, Double> vector2) {
		double innerProduct = 0;
		double norm1 = 0;
		double norm2 = 0;
		for (Key dimension : vector1.keySet()) {
			double length1 = vector1.get(dimension);
			norm1 += length1 * length1;
			if (vector2.containsKey(dimension)) {
				double length2 = vector2.get(dimension);
				innerProduct += length1 * length2;
			}
		}
		for (Key dimension : vector2.keySet()) {
			double length2 = vector2.get(dimension);
			norm2 += length2 * length2;
		}
		if (norm1 == 0 || norm2 == 0) {
			return 0;
		}
		return innerProduct / Math.sqrt(norm1 * norm2);
	}
}