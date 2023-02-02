package model;

import static java.lang.Math.exp;
import static java.lang.Math.log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import cluster.KMeans;
import data.Checkin;
import data.Sequence;
import data.SequenceDataset;
import demo.Demo;
import distribution.Gaussian;
import distribution.Multinomial;
import myutils.ArrayUtils;
import myutils.ScoreCell;
import myutils.TopKSearcher;

public class ShareHMM {
	// Fixed parameters.
	int maxIter;
	int U; // The number of users.
	int R; // The number of sequences.
	int K; // The number of latent states.
	int K_sparsity;
	int M; // The number of Gaussian components in each state.
	int D; // The number of items.
	double sparsity = 0.0; // The sparsity coefficient of the parameter c
	String underlyingDistribution;
	// weight of the sequences.
	double[] weight;
	double[] weightUserSum;
	double[] len;
	double weightSum;
	// Sequence parameters
	double[][][] alpha; // alpha[r][n][k] is for the n-th position of sequence r at state k.
	double[][][] beta; // beta[r][n][k] is for the n-th position of sequence r at state k.
	double[][] con; // con[r][n] is ln p(x_n | x_1, x_2, ... x_n-1), this is used for normalization.
	double[][][] gamma; // gamma[r][n][k]: probability that the n-th position of sequence r is at state k.
	double[][][] xi; // xi[r][j][k] is the probability that the 1st position of sequence r is state j and the 2nd position is k.
	double[][][][] rho; // rho[r][n][k][l]: probability that the n-th position of sequence r is at state k and from Gaussian component l.

	// User-specific parameters
	double[][] pi; // pi[u][i] is the prior probability at state i for user u.
	double[][][] A; // A[u][i][j] is the transition matrix, from state i to state j for user u.
	double[][][] B; // B[u][k][m] is the probability of choosing component m for state k for user u;

	// Shared parameters
	Gaussian[] geoModel; // The M components.
	Multinomial[] itemModel; // The K multinomial models for items.
	Gaussian[] temporalModel;

	// Log likelihood.
	double[][][] ll; // ll[r][n][k] is the log-likelihood p(x[r][n]|k).
	double[][] scalingFactor; // scalingFactor[r][n] is chosen from ll[r][n].
	double totalLL = 0;
	double entropy = 0;

//	// User-Sequence
//	public HashMap<Long, HashSet<Integer>> user2seqs = new HashMap<Long, HashSet<Integer>>();
//	public HashMap<Long, HashSet<Integer>> user2pnts = new HashMap<Long, HashSet<Integer>>();
//	public static List<Object> userIdList;
//	public Long[] pnts2user;
//	public int[] pnts2useridx;
	SequenceDataset data;
	
	public static int[] userNumTransitions;
	public static int[] userLenTraj;
	public static int[] userNumDifferentPlaces;
	public static double[] userEntropy;
	
	// running time start
	double elapsedTime;
	KMeans kMeans;
	
	public ShareHMM() {
	}

	public ShareHMM(int maxIter) {
		this.maxIter = maxIter;
		this.D = SequenceDataset.getD();
		
		this.R = 0;
	}

	public ShareHMM(int maxIter, String underlyingDistribution) {
		this.maxIter = maxIter;
		this.underlyingDistribution = underlyingDistribution;
		this.D = SequenceDataset.getD();
	}
	
	public void calcUserAttrInTrainSet(List<Integer>[] kMeansResults) throws IOException {
		// initial
		int[] userNumTransitions = new int[U];
		int[] userLenTraj = new int[U];
		int[] userNumDifferentPlaces = new int[U];
		double[] userEntropy = new double[U];
		// calculate the number of pnts per cluster and per user
		int[][] pntsPerClusUser = new int[kMeansResults.length][U];
		double[][] portPerClusUser = new double[kMeansResults.length][U];
		for(int k = 0; k < kMeansResults.length; k++)
			for(int i = 0; i < kMeansResults[k].size(); i++) {
				int u = data.pnts2useridx[kMeansResults[k].get(i)];
				pntsPerClusUser[k][u]++;
			}
		// calculate userNumTransitions, userLenTraj, userNumDifferentPlaces, userEntropy
		for(int u = 0; u < U; u++) {
			userNumTransitions[u] = data.user2seqs.get(data.userIdList.get(u)).size();
			for(int k = 0; k < kMeansResults.length; k++){
				userLenTraj[u] += pntsPerClusUser[k][u];
				userNumDifferentPlaces[u] += (pntsPerClusUser[k][u] > 0 ? 1 : 0);
			}
			if (userLenTraj[u] > 0)
				for(int k = 0; k < kMeansResults.length; k++){
					portPerClusUser[k][u] = (double)pntsPerClusUser[k][u] / (double)userLenTraj[u];
					if(portPerClusUser[k][u] > 0)
						userEntropy[u] += - portPerClusUser[k][u] * Math.log(portPerClusUser[k][u]);
				}
			userEntropy[u] = userEntropy[u] / Math.log(2);
		}
		// write into file
		String dir = "../result/txt/userAttr/";
		File fileDir = new File(dir);
		fileDir.mkdirs();
		FileWriter fw = new FileWriter(dir + Demo.dataset + ".txt");
		fw.write("userNumTransitions userLenTraj userNumDifferentPlaces userEntropy\n");
		for (int u = 0; u < U; u++){
			fw.write(userNumTransitions[u] + " ");
			fw.write(userLenTraj[u] + " ");
			fw.write(userNumDifferentPlaces[u] + " ");
			fw.write(userEntropy[u] + "\n");
		}
		fw.close();
	}
	
	public void loadModel() throws IOException{
		B = new double[U][K][M];
		A = new double[U][K][K];
		pi = new double[U][K];
		
		BufferedReader bufr;
		String line;
		String[] nums;
		// load the user specific params
		String dir = "../result/txt/shareHmmParas/" + Demo.dataset + "/userSpecific/";
		for (int u = 0; u < U; u++){
			// load pi
			bufr = new BufferedReader(new FileReader(dir + u + "/HMM_pi.txt"));
			for (int i = 0; i < K; i++){
				line = bufr.readLine();
				pi[u][i] = Double.parseDouble(line);
			}
			bufr.close();
			// load A
			bufr = new BufferedReader(new FileReader(dir + u + "/HMM_A.txt")); 
			for (int i = 0; i < K; i++){
				nums = bufr.readLine().split(" ");
				for (int j = 0; j < K; j++){
					A[u][i][j] = Double.parseDouble(nums[j]);
				}
			}
			bufr.close();
			// load c
			bufr = new BufferedReader(new FileReader(dir + u + "/HMM_B.txt")); 
			for (int i = 0; i < K; i++){
				nums = bufr.readLine().split(" ");
				for (int j = 0; j < M; j++){
					B[u][i][j] = Double.parseDouble(nums[j]);
				}
			}
			bufr.close();
		}
		// load underlying distributions
		bufr = new BufferedReader(new FileReader("../result/txt/shareHmmParas/" + Demo.dataset + "/shared/underlyingDistributions.txt")); 
		if(underlyingDistribution.equals("2dGaussian")){
			geoModel = new Gaussian[M];
			String meanLine, varLine;
			for (int i = 0; i < M; i++){
				meanLine = bufr.readLine();
				varLine = bufr.readLine();
				geoModel[i] = new Gaussian();
				geoModel[i].parseGaussian(meanLine, varLine);
			}
		}
		else if(underlyingDistribution.equals("multinomial")){
			itemModel = new Multinomial[M];
			for (int i = 0; i < M; i++){
				nums = bufr.readLine().split(" ");
				double[] prob = new double[D];
				for (int j = 0; j < D; j++)
					prob[j] = Double.parseDouble(nums[j]);
				itemModel[i] = new Multinomial(prob);
			}
		}
		bufr.close();
	}
	public void saveModel() throws IOException{
		FileWriter fw;
		// save the user specific params
		String dir = "../result/txt/shareHmmParas/" + Demo.dataset + "/userSpecific/";
		for (int u = 0; u < U; u++){
			File fileDir = new File(dir + u + "/");
			fileDir.mkdirs();
			// save pi
			fw = new FileWriter(dir + u + "/HMM_pi.txt"); 
			for (int i = 0; i < K; i++){
				fw.write(String.valueOf(pi[u][i]));
				fw.write("\n");
			}
			fw.close();
			// save A
			fw = new FileWriter(dir + u + "/HMM_A.txt"); 
			for (int i = 0; i < K; i++){
				for (int j = 0; j < K; j++){
					fw.write(String.valueOf(A[u][i][j]));fw.write(" ");
				}
				fw.write("\n");
			}
			fw.close();
			
			// save c
			fw = new FileWriter(dir + u + "/HMM_B.txt"); 
			for (int i = 0; i < K; i++){
				for (int j = 0; j < M; j++){
					fw.write(String.valueOf(B[u][i][j]));fw.write(" ");
				}
				fw.write("\n");
			}
			fw.close();
		}
		// save underlying distributions
		File fileDir = new File("../result/txt/shareHmmParas/" + Demo.dataset + "/shared/");
		fileDir.mkdirs(); 
		fw = new FileWriter("../result/txt/shareHmmParas/" + Demo.dataset + "/shared/underlyingDistributions.txt"); 
		if(underlyingDistribution.equals("2dGaussian")){
			for (int i = 0; i < M; i++){
				fw.write(geoModel[i].toString());
			}
		}
		else if(underlyingDistribution.equals("multinomial")){
			for (int i = 0; i < M; i++){
				fw.write(itemModel[i].toString());
			}
		}
		
		fw.close();
	}

	public void train(SequenceDataset data, int K, int M, double sparsity, String underlyingDistribution) throws IOException {
		this.sparsity = sparsity;
		if(K==1)
			K_sparsity=K+1;
		else
			K_sparsity = (int) ( (K-1)*(K-1) * sparsity ) + 1;

		this.underlyingDistribution = underlyingDistribution;
		train(data, K_sparsity, M);
	}

	public void train(SequenceDataset data, int K, int M) throws IOException {
		long start = System.currentTimeMillis();
		init(data, K, M);
		iterate(data);
		long end = System.currentTimeMillis();
		elapsedTime = (end - start) / 1000.0;
		System.out.println("The training process totally takes " + elapsedTime + "s.");
	}

	/**
	 * Step 1: initialize the geo and text models.
	 * @throws IOException 
	 */

	protected void init(SequenceDataset data, int K, int M) throws IOException {
		this.data = data;
//		 init(data, K, M, null);
		initFixedParameters(data, K, M, weight);
		initEStepParameters();
		if(Demo.loadModel)
			loadModel();
		else
			initMStepParameters(data);
		
		if(underlyingDistribution.equals("multinomial")){
			double sparsity0 = sparsity;
			int maxIter0 = maxIter;
			sparsity = Double.POSITIVE_INFINITY;
			maxIter = 3;
			iterate(data);
			sparsity = sparsity0;
			maxIter = maxIter0;
		}
		
		System.out.println("Model initialized.");
		
		
	}

	protected void init(SequenceDataset data, int K, int M, double[] weight) throws IOException {
		initFixedParameters(data, K, M, weight);
		initEStepParameters();
		initMStepParameters(data);
	}

	protected void initFixedParameters(SequenceDataset data, int K, int M, double[] weight) {
		this.R = data.size();
		this.K = K;
		this.M = M;
		// add from calcUser2seqs
		this.U = data.user2seqs.size();
		setWeight(weight);
		}

	private void setWeight(double[] weight) {
		if (weight == null) {
			this.weight = new double[R];
			for (int i = 0; i < R; i++)
				this.weight[i] = 1.0;
		} else {
			this.weight = Arrays.copyOf(weight, weight.length);
		}
		weightSum = ArrayUtils.sum(this.weight);
		weightUserSum = new double[U];
	
		for (int u = 0; u < U; u++) {
			for (int i : data.user2seqs.get(data.userIdList.get(u)))
			{
				weightUserSum[u] += this.weight[i];

			}
		}
	}

	protected void initEStepParameters() {
		ll = new double[R][2][K];
		scalingFactor = new double[R][2];
		alpha = new double[R][2][K];
		beta = new double[R][2][K];
		con = new double[R][2];
		xi = new double[R][K][K];
		gamma = new double[R][2][K];
		rho = new double[R][2][K][M];
	}

	// Initialize the paramters that need to be inferred.
	protected void initMStepParameters(SequenceDataset data) throws IOException {

		if(underlyingDistribution.equals("2dGaussian")){
			List<Integer>[] kMeansResults = runKMeans(data);
			initGeoModel(kMeansResults, data);
			calcUserAttrInTrainSet(kMeansResults);
		}
		else if(underlyingDistribution.equals("multinomial")){
			initItemModel(data);
		}
		
		initCbyRandm();
		initPibyRandm();
		initAbyRandm();
	}

	// Run k-means to initialize the params. General.
	protected List<Integer>[] runKMeans(SequenceDataset data) {
		if(underlyingDistribution.equals("2dGaussian")){
			List<Double> weights = new ArrayList<Double>();
			for (int i = 0; i < data.getGeoData().size(); i++)
				weights.add(weight[i / 2]);
			kMeans = new KMeans(M);
			return kMeans.cluster(data.getGeoData(), weights, M);
		}
		else if(underlyingDistribution.equals("multinomial")){
			List<Double> weights = new ArrayList<Double>();
			for (int i = 0; i < data.getItemData().size(); i++)
				weights.add(weight[i / 2]);
			kMeans = new KMeans(M);
			return kMeans.cluster(pntUserMat(data), weights, M); // different
		}
		else
			return null;
	}

	
	List<RealVector> pntUserMat(SequenceDataset data){
		// userNum: U; itemNum: D
		double[][] itemUser = new double[D][U];
		for (int i = 0; i < data.size(); i++) {
			itemUser[data.getItemDatum(i)][SequenceDataset.pnts2useridx[i]] += 1;
		}
		
		for (int i = 0; i < D; i++) {
			double[] itemUserSum = new double[D];
			for (int j = 0; j < U; j++) {
				itemUserSum[i] += itemUser[i][j];
			}
			for (int j = 0; j < U; j++) {
				itemUser[i][j] /= itemUserSum[i];
			}
		}
		List<RealVector> pntUser = new ArrayList<RealVector>();
		for (int i = 0; i < data.size(); i++) {
			pntUser.add(new ArrayRealVector(itemUser[data.getItemDatum(i)]));
		}
		return pntUser;
	}
	
	// initialize for geo
	protected void initCPiA(int[] dataMembership) {
		B = new double[U][K][M];
		A = new double[U][K][K];
		pi = new double[U][K];
		// user-point-cluster relationship
		double[][][] userTrans2clus = new double[U][M][2];// 用户数据点属于那个类（分前后）
		double[][] user2clus = new double[U][M]; // 用户数据点属于那个类（不分前后）
		for (int i = 0; i < dataMembership.length; i++){
			int m = dataMembership[i];
			int u = data.pnts2useridx[i];
			userTrans2clus[u][m][i%2]++;
			user2clus[u][m]++;
		}
		List<Integer>[] clustersTopKList = (List<Integer>[]) new List[U];

		// initial C: select top 5 clusters for each user
		for (int u = 0; u < U; u++){
			clustersTopKList[u] = new ArrayList<Integer>();
			TopKSearcher tks = new TopKSearcher();
			tks.init(K);
			for (int m = 0; m < M; m++)
                tks.add(new ScoreCell(m, (double) user2clus[u][m]));
            ScoreCell [] topKResults = new ScoreCell[K];
            topKResults = tks.getTopKList(topKResults);
//            List<Integer> userTopKList = new ArrayList<Integer>();
			for (int k = 0; k < K; k++){
				clustersTopKList[u].add(topKResults[k].getId());
				if(topKResults[k].getScore() > 0){
					B[u][k][topKResults[k].getId()] = 1.0;
				}
			}
		}
		// initial A and Pi
		for (int i = 0; i < dataMembership.length / 2; i++){
			int u = data.pnts2useridx[i*2];
			int m0 = dataMembership[i*2];
			int m1 = dataMembership[i*2+1];
			int k0 = clustersTopKList[u].indexOf(m0);
			int k1 = clustersTopKList[u].indexOf(m1);
			if( k0 != -1 && k1 != -1){
				A[u][k0][k1]++;
				pi[u][k0]++;
			}
		}
		for (int u = 0; u < U; u++){
			normalize(pi[u]);
			for (int k = 0; k < K; k++){
				normalize(A[u][k]);
				normalize(B[u][k]);
			}
		}
	}
	
	void normalize(double[] vector){
		double sum = 0;
		for (int i = 0; i < vector.length; i++)
			sum += vector[i];
		if (sum == 0)
			for (int i = 0; i < vector.length; i++)
				vector[i] = 1.0 / vector.length;
		else
			for (int i = 0; i < vector.length; i++)
				vector[i] = vector[i] / sum;
	}
	
	// numDataPoints is 2R.
	protected void initCbyRandm() {
		B = new double[U][K][M];
		Random random = new Random(2);//2
		for (int u = 0; u < U; u++) {
			for (int j = 0; j < K; j++) {
				double sum = 0;
				for (int m = 0; m < M; m++) {
					B[u][j][m] = random.nextDouble();
					sum += B[u][j][m];
				}
				for (int m = 0; m < M; m++)
					B[u][j][m] = B[u][j][m] / sum;
			}
		}
	}
	
	protected void initPibyRandm() {
		pi = new double[U][K];
		Random random = new Random(1);//1
		for (int u = 0; u < U; u++) {
			double sum = 0;
			for (int i = 0; i < K; i++) {
				pi[u][i] = random.nextDouble();
				sum += pi[u][i];
			}
			for (int i = 0; i < K; i++)
				pi[u][i] = pi[u][i] / sum;
		}
	}

	protected void initAbyRandm() {
		A = new double[U][K][K];
		Random random = new Random(2);//2
		for (int u = 0; u < U; u++) {
			for (int j = 0; j < K; j++) {
				double sum = 0;
				for (int i = 0; i < K; i++) {
					A[u][j][i] = random.nextDouble();
					sum += A[u][j][i];
				}
				for (int i = 0; i < K; i++)
					A[u][j][i] = A[u][j][i] / sum;
			}
		}
	}

	// Find the kmeans membership for the 2*R places
	protected int[] findMemebership(List<Integer>[] kMeansResults) {
		int[] dataMembership = new int[2 * R];
		for (int clusterId = 0; clusterId < K; clusterId++) {
			List<Integer> clusterDataIds = kMeansResults[clusterId];
			for (int dataId : clusterDataIds)
				dataMembership[dataId] = clusterId;
		}
		return dataMembership;
	}

	// Initialize the geo model and c
	protected void initGeoModel(List<Integer>[] kMeansResults, SequenceDataset data) {
		this.geoModel = new Gaussian[M]; // K states, each having M components
		for (int m = 0; m < M; m++) {
			List<RealVector> clusterData = new ArrayList<RealVector>();
			List<Double> clusterWeights = new ArrayList<Double>();
			List<Integer> dataIds = kMeansResults[m];
			for (int dataId : dataIds) {
				clusterData.add(data.getGeoDatum(dataId));
				clusterWeights.add(weight[dataId / 2]);
			}
			geoModel[m] = new Gaussian();
			System.out.println(m + "fd" + clusterData.get(0).getDimension());
//			System.out.println(clusterData.get(0).getDimension()+ "fd" );			
			geoModel[m].fit(clusterData, clusterWeights);
		}
	}
	
	protected void initItemModel(SequenceDataset data) {
		this.itemModel = new Multinomial[M];
		Random random = new Random(18);//1
		for (int m = 0; m < M; m++)
		{
			double[] prob = new double[D];
			for (int d = 0; d < D; d++)
				prob[d] = random.nextDouble();
			itemModel[m] = new Multinomial(prob);
		}
	}
	
	protected void initItemModel(RealVector[] means, List<Integer>[] kMeansResults, SequenceDataset data){
		double[][] clusItem = new double[M][D];
		this.itemModel = new Multinomial[M];
		for (int m = 0; m < M; m++)
		{
			for (int i : kMeansResults[m])
				clusItem[m][data.getItemDatum(i)] += 1; 
			itemModel[m] = new Multinomial(clusItem[m]);
		}
	}
	
	// Initialize the temporal model and c
	protected void initTemporalModel(List<Integer>[] kMeansResults, SequenceDataset data) {
		this.temporalModel = new Gaussian[K]; // K states, each having M
												// components
		for (int k = 0; k < K; k++) {
			List<RealVector> clusterData = new ArrayList<RealVector>();
			List<Double> clusterWeights = new ArrayList<Double>();
			List<Integer> dataIds = kMeansResults[k];
			for (int dataId : dataIds) {
				clusterData.add(data.getTemporalDatum(dataId));
				clusterWeights.add(weight[dataId / 2]);
			}
			temporalModel[k] = new Gaussian();
			temporalModel[k].fit(clusterData, clusterWeights);
		}
	}

	/**
	 * Step 2: iterate over the e-step and m-step.
	 * @throws IOException 
	 */
	protected void iterate(SequenceDataset data) throws IOException {
		double prevLL = totalLL;
		double prevEntropy = entropy;
		for (int iter = 0; iter < maxIter; iter++) {
			//System.out.println("sparsity = " + sparsity);
			

			
			eStep(data);
			
			mStep(data);

			calcTotalLL();
					
			
			writeObj();
	
			
			if (Double.isNaN(totalLL))
				System.out.println("totalLL is NAN");
			if (Demo.printLL)
				System.out.println("ShareHMM finished iteration " + iter + ". Log-likelihood:" + totalLL 
						+ ". Objective:" + (totalLL + entropy) + "." );
			if (Math.abs(totalLL + entropy - prevLL - prevEntropy) <= 0.01)
				break;
			prevLL = totalLL;
			prevEntropy = entropy;
			if(Demo.saveModel)
				saveModel();
		}
	}

	void writeObj() throws IOException {
		String dir = "../result/txt/objCnvg/";
		String filename = dir + Demo.dataset + ".txt";
//		File fileDir = new File(dir);
//		fileDir.mkdirs();
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename, true)));
//		FileWriter fw = new FileWriter(dir + Demo.dataset + ".txt");
		out.write(totalLL + " ");
		out.write(entropy + " ");
		out.write((totalLL + entropy) + "\n");
		out.close();
	}
	
	
	/**
	 * Step 2.1: learning the parameters using EM: E-Step.
	 */
	protected void eStep(SequenceDataset data) {
		long programStart = System.currentTimeMillis();
		calcLL(data);
		long programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("calcLL() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		scaleLL();
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("scaleLL() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		calcAlpha();
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("calcAlpha() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		calcBeta();
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("calcBeta() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		calcGamma();
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("calcGamma() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		calcXi();
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("calcXi() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		calcRho(data);
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("calcRho() takes " + (programEnd - programStart) / 1000.0 + "s.");

	}

	// Compute the log likelihood.
	protected void calcLL(SequenceDataset data) {
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				for (int k = 0; k < K; k++) {
					int u = data.pnts2useridx[r * 2];
					switch(underlyingDistribution){
					case("2dGaussian"): ll[r][n][k] = calcLLState(data.getGeoDatum(2 * r + n), k, u); break;
					case("multinomial"): ll[r][n][k] = calcLLState(data.getItemDatum(2 * r + n), k, u); break;
					}
				}
	}

	protected void scaleLL() {
		// Find the scaling factors.
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				scalingFactor[r][n] = ArrayUtils.max(ll[r][n]);
		// Scale the log-likelihood.
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				for (int k = 0; k < K; k++)
					ll[r][n][k] -= scalingFactor[r][n];
	}

	protected void calcAlpha() {
		// Compute alpha[r][0][k], in the log domain!
		for (int r = 0; r < R; r++)
			for (int k = 0; k < K; k++) {
				alpha[r][0][k] = log(pi[data.pnts2useridx[r * 2]][k]) + ll[r][0][k];
				if (Double.isNaN(alpha[r][0][k]))
					System.out.println("alpha[r][0][k] is NaN");
			}
		// Compute con[r][0], namely ln p(x_0)
		for (int r = 0; r < R; r++) {
			con[r][0] = ArrayUtils.sumExpLog(alpha[r][0]);
			if (Double.isNaN(con[r][0]))
				System.out.println("con[r][0] is NaN");
		}
		// Normalize alpha[r][0][k]
		for (int r = 0; r < R; r++)
			ArrayUtils.logNormalize(alpha[r][0]);
		// Compute alpha[r][1][k], again in the log domain.
		for (int r = 0; r < R; r++) {
			for (int k = 0; k < K; k++) {
				alpha[r][1][k] = ll[r][1][k];
				double sum = 1e-200;
				for (int j = 0; j < K; j++) {
					if (Double.isNaN(alpha[r][0][j]))
						System.out.println("alpha[r][0][j] is NaN");
					if (Double.isNaN(A[data.pnts2useridx[r * 2]][j][k]))
						System.out.println("A[pnts2useridx[r*2]][j][k] is NaN");
					sum += alpha[r][0][j] * A[data.pnts2useridx[r * 2]][j][k];
					if (Double.isNaN(sum))
						System.out.println("sum is NaN");
					if (sum == 0)
						System.out.println("sum is 0");
				}
				alpha[r][1][k] += log(sum);
				if (Double.isNaN(alpha[r][1][k]))
					System.out.println("alpha[r][1][k] is NaN");
			}
		}
		// Compute con[r][1], namely ln p(x_1 | x_0)
		for (int r = 0; r < R; r++) {
			con[r][1] = ArrayUtils.sumExpLog(alpha[r][1]);
			if (Double.isNaN(con[r][1]))
				System.out.println("con[r][1] is NaN");
		}
		// Normalize alpha[r][1][k]
		for (int r = 0; r < R; r++)
			ArrayUtils.logNormalize(alpha[r][1]);
	}

	protected void calcBeta() {
		// Compute beta[r][1][k]
		for (int r = 0; r < R; r++)
			for (int k = 0; k < K; k++)
				beta[r][1][k] = 1.0;
		// Compute beta[r][0][k]
		for (int r = 0; r < R; r++) {
			for (int k = 0; k < K; k++) {
				double sum = 0;
				for (int j = 0; j < K; j++) {
					if (A[data.pnts2useridx[r * 2]][k][j] == 0)
						sum += 0;
					else if (ll[r][1][j] - con[r][1] >= 500)
						sum += A[data.pnts2useridx[r * 2]][k][j] * 1e200;
					else
						sum += exp(ll[r][1][j] - con[r][1]) * A[data.pnts2useridx[r * 2]][k][j];
				}
				beta[r][0][k] = sum;
			}
		}
	}

	protected void calcGamma() {
		for (int r = 0; r < R; r++)
			for (int n = 0; n < 2; n++)
				for (int k = 0; k < K; k++)
					gamma[r][n][k] = alpha[r][n][k] * beta[r][n][k];
	}

	protected void calcXi() {
		for (int r = 0; r < R; r++)
			for (int j = 0; j < K; j++)
				for (int k = 0; k < K; k++)
					xi[r][j][k] = alpha[r][0][j] * exp(ll[r][1][k] - con[r][1]) * A[data.pnts2useridx[r * 2]][j][k]
							* beta[r][1][k];
	}

	protected void calcRho(SequenceDataset data) {
		if(underlyingDistribution.equals("2dGaussian")){
			for (int r = 0; r < R; r++) {
				for (int n = 0; n < 2; n++) {
					RealVector v = data.getGeoDatum(2 * r + n);
					for (int k = 0; k < K; k++) {
						for (int m = 0; m < M; m++) {
							int u = data.pnts2useridx[r * 2];
							rho[r][n][k][m] = calcGeoLLComponent(v, k, m, u); // Log domain.
						}
						ArrayUtils.logNormalize(rho[r][n][k]); // Transform to normal domain.
						for (int m = 0; m < M; m++)
							rho[r][n][k][m] = gamma[r][n][k] * rho[r][n][k][m];
					}
				}
			}
		}
		else if(underlyingDistribution.equals("multinomial")){
			for (int r = 0; r < R; r++) {
				for (int n = 0; n < 2; n++) {
					int v = data.getItemDatum(2 * r + n);
					for (int k = 0; k < K; k++) {
						for (int m = 0; m < M; m++) {
							int u = data.pnts2useridx[r * 2];
							rho[r][n][k][m] = calcItemLLComponent(v, k, m, u); // Log domain.
						}
						ArrayUtils.logNormalize(rho[r][n][k]); // Transform to normal domain.
						for (int m = 0; m < M; m++)
							rho[r][n][k][m] = gamma[r][n][k] * rho[r][n][k][m];
					}
				}
			}
		}
		
	}

	/**
	 * Step 2.2: learning the parameters using EM: M-Step.
	 */
	protected void mStep(SequenceDataset data) {
		long programStart = System.currentTimeMillis();
		updatePi();
		long programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("updatePi() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		updateA();
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("updateA() takes " + (programEnd - programStart) / 1000.0 + "s.");

		programStart = System.currentTimeMillis();
		updateC();
		programEnd = System.currentTimeMillis();
		if (Demo.printDetail)
			System.out.println("updateC() takes " + (programEnd - programStart) / 1000.0 + "s.");

		if(underlyingDistribution.equals("2dGaussian")){
			programStart = System.currentTimeMillis();
			updateGeoModel(data);
			programEnd = System.currentTimeMillis();
			if (Demo.printDetail)
				System.out.println("updateGeoModel(data); takes " + (programEnd - programStart) / 1000.0 + "s.");
		}
		else if(underlyingDistribution.equals("multinomial")){
			programStart = System.currentTimeMillis();
			updateItemModel(data);
			programEnd = System.currentTimeMillis();
			if (Demo.printDetail)
				System.out.println("updateItemModel(data); takes " + (programEnd - programStart) / 1000.0 + "s.");
		}

		// updateTextModel(data);
		// updateGeoModel(data);
		// updateTemporalModel(data);
	}

	protected void updatePi() {
		double[][] numerator = new double[U][K];
		for (int k = 0; k < K; k++) {
			for (int r = 0; r < R; r++) {
				int u = data.pnts2useridx[r * 2];
				numerator[u][k] += gamma[r][0][k] * weight[r];
			}
		}
		for (int u = 0; u < U; u++) {
			for (int k = 0; k < K; k++) {
				// whether to * 2
				pi[u][k] = numerator[u][k] / (weightUserSum[u]);
			}
		}
	}

	protected void updateA() {
		double min = 1.0e-8 / K;
		double[][] denominator = new double[U][K];
		for (int u = 0; u < U; u++)
			for (int k = 0; k < K; k++) {
				denominator[u][k] += min * K;
			}
		for (int j = 0; j < K; j++) {
			for (int r = 0; r < R; r++)
				for (int k = 0; k < K; k++) {
					int u = data.pnts2useridx[r * 2];
					denominator[u][j] += weight[r] * xi[r][j][k];
				}
			double[][] numerator = new double[U][K];
			for (int u = 0; u < U; u++)
				for (int k = 0; k < K; k++)
					numerator[u][k] += min;
			for (int k = 0; k < K; k++) {
				for (int r = 0; r < R; r++) {
					int u = data.pnts2useridx[r * 2];
					numerator[u][k] += weight[r] * xi[r][j][k];
				}
				for (int u = 0; u < U; u++) {
//					if (denominator[u][j] != 0)
					A[u][j][k] = numerator[u][k] / denominator[u][j];
					if (Double.isNaN(A[u][j][k])) {
						System.out.println("numerator[u][k] = " + numerator[u][k] + "; denominator[u][j] = " + denominator[u][j]);
						System.out.println("A[u][j][k] is NaN. ");
					}
				}
			}
		}
	}

	protected void updateGeoModel(SequenceDataset data) {
		for (int m = 0; m < M; m++) {
			List<Double> weights = new ArrayList<Double>();
			for (int r = 0; r < R; r++)
				for (int n = 0; n < 2; n++) {
					double w = 0;
					for (int k = 0; k < K; k++)
						w += weight[r] * rho[r][n][k][m];
					weights.add(w);
				}
			geoModel[m].fit(data.getGeoData(), weights);
		}
	}

	protected void updateItemModel(SequenceDataset data) {
		// updateC();
		for (int m = 0; m < M; m++) {
			List<Double> weights = new ArrayList<Double>();
			for (int r = 0; r < R; r++)
				for (int n = 0; n < 2; n++) {
					double w = 0;
					for (int k = 0; k < K; k++)
						w += weight[r] * rho[r][n][k][m];
					weights.add(w);
				}
			itemModel[m].fit(D, data.getItemData(), weights);
		}
	}
	
	protected void updateC() {
		len = new double[U];
		double[][] denominator = new double[U][K];
		for (int k = 0; k < K; k++) {
			for (int r = 0; r < R; r++)
				for (int n = 0; n < 2; n++) {
					int u = data.pnts2useridx[r * 2];
					denominator[u][k] += weight[r] * gamma[r][n][k];
					len[u] += 2;
				}
			double[][][] numerator = new double[U][K][M];
			for (int m = 0; m < M; m++) {
				for (int r = 0; r < R; r++)
					for (int n = 0; n < 2; n++) {
						int u = data.pnts2useridx[r * 2];
						numerator[u][k][m] += weight[r] * rho[r][n][k][m];
					}
			}
			for (int u = 0; u < U; u++) {
				SparseEstimator sparseEstimator = new SparseEstimator(sparsity); // sparsity * len[u]
				RealVector sumGamma = new ArrayRealVector(M);
				for (int m = 0; m < M; m++) 
					sumGamma.setEntry(m, numerator[u][k][m]);
				if(sumGamma.getL1Norm() > 0){ // update when the sum of sumGamma is not zero
					RealVector alpha = sparseEstimator.estimate(sumGamma);
					for (int m = 0; m < M; m++) {
						B[u][k][m] = alpha.getEntry(m);// numerator[u][k][m] / denominator[u][k];
					}
				}
			}
		}
	}

	protected void updateTemporalModel(SequenceDataset data) {
		for (int k = 0; k < K; k++) {
			List<Double> weights = new ArrayList<Double>();
			for (int r = 0; r < R; r++)
				for (int n = 0; n < 2; n++)
					weights.add(weight[r] * gamma[r][n][k]);
			temporalModel[k].fit(data.getTemporalData(), weights);
		}
	}

	/**
	 * Functions for computing probabilities
	 */
	// Calc the likelihood that the given data is generated from state k
//	protected double calcLLState(RealVector geoDatum, int k, int u) {
//		return calcLLState(geoDatum, k, u, false);
//	}

//	protected double calcLLState(RealVector geoDatum, int k, int u, boolean isTest) {
//		double geoProb = calcGeoLLState(geoDatum, k, u);
//		return geoProb;
//	}

	// Calc the probability that v is generated from the gmm of state k.
	protected double calcGeoLLState(RealVector v, int k, int u) {
		double[] lnProb = new double[M];
		for (int m = 0; m < M; m++)
			lnProb[m] = calcGeoLLComponent(v, k, m, u);
		double maxLnProb = ArrayUtils.max(lnProb);
		for (int m = 0; m < M; m++)
			lnProb[m] -= maxLnProb;
		double sum = 0;
		for (int m = 0; m < M; m++)
			sum += exp(lnProb[m]);
		return maxLnProb + log(sum);
	}
	
	protected double calcItemLLState(int v, int k, int u) {
		double[] lnProb = new double[M];
		for (int m = 0; m < M; m++)
			lnProb[m] = calcItemLLComponent(v, k, m, u);
		double maxLnProb = ArrayUtils.max(lnProb);
		for (int m = 0; m < M; m++)
			lnProb[m] -= maxLnProb;
		double sum = 0;
		for (int m = 0; m < M; m++)
			sum += exp(lnProb[m]);
		return maxLnProb + log(sum);
	}

	// Compute the prob that v is generated from the m-th component of state k.
	protected double calcGeoLLComponent(RealVector v, int k, int m, int u) {
		double prior = B[u][k][m];
		double logGeoProb = geoModel[m].calcLL(v);
		return log(prior) + logGeoProb;
	}
	
	protected double calcItemLLComponent(int v, int k, int m, int u) {
		double prior = B[u][k][m];
		double logItemProb = itemModel[m].calcLL(v);
		return log(prior) + logItemProb;
	}
	

	protected void calcTotalLL() {
		totalLL = 0;
		for (int r = 0; r < R; r++) {
			double hmmLL = 0;
			for (int n = 0; n < 2; n++) {
				if (Double.isNaN(con[r][n]))
					System.out.println("con[r][n] is NaN");
				if (Double.isNaN(scalingFactor[r][n]))
					System.out.println("scalingFactor[r][n] is NaN");
				hmmLL += con[r][n] + scalingFactor[r][n];
				if (Double.isNaN(hmmLL))
					System.out.println("hmmLL is NaN");
			}
			if (Double.isNaN(weight[r]))
				System.out.println("weight[r] is NAN, r = " + r);
			totalLL += weight[r] * hmmLL;
			if (Double.isNaN(totalLL))
				System.out.println("totalLL is NAN");
		}
		
		entropy = 0;
		for (int u =0; u < U; u++)
			for (int i = 0;  i < K; i++)
				for (int j = 0;  j < M; j++)
					if(B[u][i][j] != 0)
						entropy += B[u][i][j] * Math.log(B[u][i][j]); // * len[u];
		entropy = entropy * sparsity;
	}

	/**
	 * Compute the ll of a test sequence.
	 */
	public double calcItemLL(List<Integer> items, int u) {
		return calcItemLL(items, u, false);
	}

	public double calcItemLL(List<Integer> items, int u, boolean isTest) {
		double[][] ll = new double[2][K]; // ll[n][k] is the log-likelihood p(x[n]|k).
		double[] scalingFactor = new double[2]; // scalingFactor[n] is chosen from ll[n].
		double[][] alpha = new double[2][K]; // alpha[n][k] is for the n-th position of sequence r at state k.
		double[] con = new double[2]; // con[n] is ln p(x_n | x_1, x_2, ... x_n-1), this is used for normalization.
		// calc LL
		for (int n = 0; n < 2; n++)
			for (int k = 0; k < K; k++)
				ll[n][k] = calcLLState(items.get(n), k, u);
		// Find the scaling factors.
		for (int n = 0; n < 2; n++)
			scalingFactor[n] = ArrayUtils.max(ll[n]);
		// Scale the log-likelihood.
		for (int n = 0; n < 2; n++)
			for (int k = 0; k < K; k++)
				ll[n][k] -= scalingFactor[n];
		// Compute alpha[0][k], in the log domain!
		for (int k = 0; k < K; k++) {
			alpha[0][k] = log(pi[u][k]) + ll[0][k];
			if (Double.isNaN(alpha[0][k]))
				System.out.println("alpha[0][k] is NAN. k = " + k);
		}
		// Compute con[0], namely ln p(x_0)
		con[0] = ArrayUtils.sumExpLog(alpha[0]);
		// Normalize alpha[0][k]
		ArrayUtils.logNormalize(alpha[0]);
		// Compute alpha[1][k], again in the log domain.
		for (int k = 0; k < K; k++) {
			alpha[1][k] = ll[1][k];
			double sum = 1e-200;
			for (int j = 0; j < K; j++) {
				sum += alpha[0][j] * A[u][j][k];
			}
			alpha[1][k] += log(sum);
			if (Double.isNaN(alpha[1][k]))
				System.out.println("alpha[l][k] is NAN. k = " + k);
		}
		// Compute con[1], namely ln p(x_1 | x_0)
		con[1] = ArrayUtils.sumExpLog(alpha[1]);
		// the result ll.
		double hmmLL = 0;
		for (int n = 0; n < 2; n++)
			hmmLL += con[n] + scalingFactor[n];
		return hmmLL;
	}
	
	public double calcGeoLL(List<RealVector> geo, int u) {
		return calcGeoLL(geo, u, false);
	}

	public double calcGeoLL(List<RealVector> geo, int u, boolean isTest) {
		double[][] ll = new double[2][K]; // ll[n][k] is the log-likelihood p(x[n]|k).
		double[] scalingFactor = new double[2]; // scalingFactor[n] is chosen from ll[n].
		double[][] alpha = new double[2][K]; // alpha[n][k] is for the n-th position of sequence r at state k.
		double[] con = new double[2]; // con[n] is ln p(x_n | x_1, x_2, ... x_n-1), this is used for normalization.
		// calc LL
		for (int n = 0; n < 2; n++)
			for (int k = 0; k < K; k++)
				ll[n][k] = calcLLState(geo.get(n), k, u);
		// Find the scaling factors.
		for (int n = 0; n < 2; n++)
			scalingFactor[n] = ArrayUtils.max(ll[n]);
		// Scale the log-likelihood.
		for (int n = 0; n < 2; n++)
			for (int k = 0; k < K; k++)
				ll[n][k] -= scalingFactor[n];
		// Compute alpha[0][k], in the log domain!
		for (int k = 0; k < K; k++) {
			alpha[0][k] = log(pi[u][k]) + ll[0][k];
			if (Double.isNaN(alpha[0][k]))
				System.out.println("alpha[0][k] is NAN. k = " + k);
		}
		// Compute con[0], namely ln p(x_0)
		con[0] = ArrayUtils.sumExpLog(alpha[0]);
		// Normalize alpha[0][k]
		ArrayUtils.logNormalize(alpha[0]);
		// Compute alpha[1][k], again in the log domain.
		for (int k = 0; k < K; k++) {
			alpha[1][k] = ll[1][k];
			double sum = 1e-200;
			for (int j = 0; j < K; j++) {
				sum += alpha[0][j] * A[u][j][k];
			}
			alpha[1][k] += log(sum);
			if (Double.isNaN(alpha[1][k]))
				System.out.println("alpha[l][k] is NAN. k = " + k);
		}
		// Compute con[1], namely ln p(x_1 | x_0)
		con[1] = ArrayUtils.sumExpLog(alpha[1]);
		// the result ll.
		double hmmLL = 0;
		for (int n = 0; n < 2; n++)
			hmmLL += con[n] + scalingFactor[n];
		return hmmLL;
	}

	protected double calcLLState(RealVector geoDatum, int k, int u) {
		return calcGeoLLState(geoDatum, k, u);
	}
	
	protected double calcLLState(int itemDatum, int k, int u) {
		return calcItemLLState(itemDatum, k, u);
	}

	public double calcSeqScore(Sequence seq, int u) {
		Checkin startPlace = seq.getCheckin(0);
		Checkin endPlace = seq.getCheckin(1);
		double prob = 0;
		if(underlyingDistribution.equals("2dGaussian")){
			List<RealVector> geo = new ArrayList<RealVector>();
			geo.add(startPlace.getLocation().toRealVector());
			geo.add(endPlace.getLocation().toRealVector());
			return calcGeoLL(geo, u);
		}
		else if(underlyingDistribution.equals("multinomial")){
			List<Integer> items = new ArrayList<Integer>();
			items.add(startPlace.getItemId());
			items.add(endPlace.getItemId());
			prob = calcItemLL(items, u);
		}
		return prob;
	}
	
	// predicting item don't need to generate 
	public int[] nextItemTopK(int itemId, int userId, int K) {
		// initial top k index and the corresponding score (setted as NEGATIVE_INFINITY)
		int[] nextTopK = new int[K];
		double[] nextTopKscore = new double[K];
		for (int i = 0; i < nextTopKscore.length;i++){
			nextTopKscore[i] = Double.NEGATIVE_INFINITY;
		}
		// construct length-2 seqs: items
		List<Integer> items = new ArrayList<Integer>();
		items.add(itemId);
		items.add(0);
		// go through all the items and select the top K results stored in nextTopK and nextTopK
		for (int i = 0; i < SequenceDataset.getD(); i++){
			// change the 2nd item and calculate the score for the seq
			items.set(1, i);
			double score = calcItemLL(items, userId);
			// check whether the new item's score is larger than the score in the topK list
			for(int j = 0; j < K; j++) // score is sorted descendingly
				if(score > nextTopKscore[j]){
					for (int k = K-1; k > j; k--){
						nextTopK[k] = nextTopK[k-1];
						nextTopKscore[k] = nextTopKscore[k-1];
					}
					nextTopK[j] = i;
					nextTopKscore[j] = score;
					break;
				}
		}
		return nextTopK;
	}

	/**
	 * Methods for the ensemble of HMM.
	 * @throws IOException 
	 */
	// don't need to return LL, since I noticed the LL is stored in "totalLL" and can be accessed any time
	public void train(SequenceDataset data, int K, int M, double[] seqsFracCount) throws IOException {
		init(data, K, M, seqsFracCount);
		iterate(data);
	}

	// don't need to return LL, since I noticed the LL is stored in "totalLL" and can be accessed any time
	public void update(SequenceDataset data, double[] seqsFracCount) throws IOException {
		setWeight(seqsFracCount);
		iterate(data);
	}

	public double getTotalLL() {
		return totalLL;
	}
}
