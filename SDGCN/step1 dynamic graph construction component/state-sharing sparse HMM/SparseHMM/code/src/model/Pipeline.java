package model;

import java.io.IOException;
import java.util.List;

import org.apache.commons.math3.linear.RealVector;

import data.SequenceDataset;
import demo.Demo;
import distribution.Gaussian;
import distribution.Multinomial;

public class Pipeline extends ShareHMM {
	double[] pnt2clus;

	public Pipeline(int maxIter) {
		super(maxIter);
		this.D = SequenceDataset.getD();
	}
	
	public Pipeline(int maxIter, String underlyingDistribution) {
		super(maxIter);
		this.underlyingDistribution = underlyingDistribution;
		this.D = SequenceDataset.getD();
	}

	public void train(SequenceDataset data, int K, int M) throws IOException {
		long start = System.currentTimeMillis();
		System.out.println("Initialization begins...");
		init(data, K, M);
		System.out.println("Initialization finished...");
//		iterate(data);
		long end = System.currentTimeMillis();
		elapsedTime = (end - start) / 1000.0;
		System.out.println("The pipeline (only inluding initialization) totally takes " + elapsedTime + "s.");
	}
	
	// Initialize the parameters that need to be inferred.
		protected void initMStepParameters(SequenceDataset data) throws IOException {
			System.out.println("Kmeans begins...");
			List<Integer>[] kMeansResults = runKMeans(data);
			System.out.println("Kmeans finished...");
			if(underlyingDistribution.equals("2dGaussian")){
				calcUserAttrInTrainSet(kMeansResults);
				initGeoModel(kMeansResults, data);
			}else if(underlyingDistribution.equals("multinomial")){
				initItemModel(kMeans.getMeans(), kMeansResults, data);
			}
			int[] dataMembership = findMemebership(kMeansResults);
			// pnt user
			initCPiA(dataMembership);
		}
}