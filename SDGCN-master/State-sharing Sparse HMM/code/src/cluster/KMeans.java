package cluster;

import myutils.ArrayUtils;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.*;

public class KMeans{

	int maxIter;
	Random r = new Random();
	RealVector [] prevMean = null;
	RealVector [] mean = null;
	List<Integer> [] clusters = null;

	int nCluster;
	List<RealVector> data = null;
	List<Double> weights = null;

	public KMeans(int maxIter) {
		this.maxIter = maxIter;
		r.setSeed(100);
	}

	public List<Integer> [] cluster(List<RealVector> data, List<Double> weights, int K) {
		if(data.size() < K) {
			System.out.println("Warning: fewer data points than the cluster number.");
			this.nCluster = data.size();
		}
		initialize(data, weights, K);
		for (int iteration = 0; iteration < maxIter; iteration++) {
			computeMean();
			if( !hasMeanChanged() )
				break;
			assignToClusters();
		}
		return clusters;
	}

	// cluster based on both attribute 1 and attribute 2.
	public List<Integer> [] cluster(List<RealVector> attr1, List<RealVector> attr2, List<Double> weights, int K) {
		if(attr1.size() != attr2.size()) {
			System.out.println("Attribute length not equal.");
			return null;
		}
		List<RealVector> data = mergeData(attr1, attr2);
		return cluster(data, weights, K);
	}

    // normalize and merge the attributes.
	private List<RealVector> mergeData(List<RealVector> attr1, List<RealVector> attr2) {
		List<RealVector> normal1 = normalize(attr1);
		List<RealVector> normal2 = normalize(attr2);
		List<RealVector> data = new ArrayList<RealVector>();
		for (int i = 0; i < normal1.size(); i++) {
			RealVector rv = new ArrayRealVector((ArrayRealVector)normal1.get(i), normal2.get(i));
			data.add(rv);
		}
		return data;
	}

	private List<RealVector> normalize(List<RealVector> attr) {
		// clone the input data.
		List<RealVector> normalizedData = new ArrayList<RealVector>();
		for (RealVector rv : attr)
			normalizedData.add(rv.copy());
		// normalize
		int dim = attr.get(0).getDimension();
		for (int i = 0; i < dim; i++) {
			double [] dimNormalData = new double[attr.size()];
			for (int j = 0; j < attr.size(); j++) {
				dimNormalData[j] = attr.get(j).getEntry(i);
			}
			ArrayUtils.normalizeZeroOne(dimNormalData);
			for (int j = 0; j < attr.size(); j++) {
				normalizedData.get(j).setEntry(i, dimNormalData[j]);
			}
		}
		return normalizedData;
	}


	private void initialize(List<RealVector> data, List<Double> weights, int K) {
		this.data = data;
		this.weights = weights;
		nCluster = K;
		prevMean = new RealVector[K];
		mean = getRandomMean();
		clusters = new List [K];
		for(int i=0; i<K; i++) {
			clusters[i] = new ArrayList<Integer>();
		}
		assignToClusters();
	}


	private RealVector [] getRandomMean() {
		Set<RealVector> randomPoints = new HashSet<RealVector>();
		int n = data.size();  // Total number of data points.
//		System.out.println("现在打印" + n);
		int[] completeArray = new int[n];  // The array of indices.
		for(int i=0; i<n; i++) {
			completeArray[i] = i;
		}
		int bound = n;
//		System.out.println("现在打印" + bound);	
//		System.out.println("现在打印" + nCluster);	
//		System.out.println("现在打印" + randomPoints.size());			
		while (randomPoints.size() < nCluster) {
			int randNum = r.nextInt(bound); //generate a random integer between 0~bound-1
			randomPoints.add(data.get(randNum));
//			System.out.println("产生数据" + data.get(randNum));	
			completeArray[randNum] = completeArray[ bound-1 ];
			bound --;
			if(bound==1)
			{
				System.out.println("随机数" + randNum);				
				System.out.println("新边界" + bound);	
				System.out.println("总随机数" + randomPoints.size());	
			}
			
		}
		return randomPoints.toArray(new RealVector[randomPoints.size()]);
	}


	private void assignToClusters() {
		for(int i=0; i<clusters.length; i++)
			clusters[i] = new ArrayList<Integer>();
		for(int i=0; i<data.size(); i++) {
			int assignId = getNearestCluster(data.get(i));
			clusters[ assignId ].add(i);  // assign data i
		}
	}


	private int getNearestCluster(RealVector p) {
		int result = 0;
		double minDist = mean[0].getDistance(p);
		for(int i=1; i<nCluster; i++) {
			double dist = mean[i].getDistance(p);
			if(dist <= minDist) {
				minDist = dist;
				result = i;
			}
		}
		return result;
	}


	private void computeMean() {
		// before computing, store current version of mean into previous mean
		for(int i=0; i<nCluster; i++)
			prevMean[i] = mean[i].copy();
		for(int i=0; i<nCluster; i++) {
			double sumWeight = 0;
			mean[i] = new ArrayRealVector( prevMean[i].getDimension() );
			List<Integer> dataIds = clusters[i];
			for(Integer dataId : dataIds) {
				double weight = weights.get(dataId);
				mean[i] = mean[i].add(data.get(dataId).mapMultiply(weight));
				sumWeight += weight;
			}
			mean[i].mapDivideToSelf(sumWeight);
		}
	}


	private boolean hasMeanChanged() {
		for(int i=0; i<nCluster; i++) {
			if(!prevMean[i].equals(mean[i]))
				return true;
		}
		return false;
	}
	
	public RealVector[] getMeans(){
		return mean;
	}

	public static void main(String [] args) {
		RealVector rv1 = new ArrayRealVector(new double []{10, -5});
		RealVector rv2 = new ArrayRealVector(new double []{10, -5});
		RealVector rv3 = new ArrayRealVector(new double []{10, -5});
		RealVector rv4 = new ArrayRealVector(new double []{10, -4});
		List<RealVector> data = new ArrayList<RealVector>();
		data.add(rv1);
		data.add(rv2);
		data.add(rv3);
		data.add(rv4);
		List<Double> weights = new ArrayList<Double>();
		for(int i=0; i<4; i++)
			weights.add(1.0);
		KMeans kMeans = new KMeans(100);
		List<Integer> [] results = kMeans.cluster(data, weights, 2);
		for (int i=0; i<results.length; i++) {
			System.out.println(results[i]);
		}
	}

}
