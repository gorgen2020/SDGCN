package demo;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import JavaExtractor.utils;

//import com.sun.xml.internal.bind.v2.model.runtime.RuntimeValuePropertyInfo;

import data.PredictionDataset;
import data.SequenceDataset;
import model.EHMM;
import model.HMM;
import model.Pipeline;
import model.ShareHMM;
import predict.DistancePredictor;
import predict.EHMMPredictor;
import predict.HMMPredictor;
import predict.ShareHMMPredictor;

/**
 * The main file for evaluating the models. Created by Liang Guojun on 10/9/21.
 */
public class Demo {
	
	public static String city = "beijing"; //set the training city
	public static String option = "default"; //set the option
	
	public static String SAVE_PATH = "C:\\Users\\gorgen\\Desktop\\S3HMM\\result.txt";
	public static String ACC_SAVE_PATH = "C:\\Users\\gorgen\\Desktop\\S3HMM\\result.txt";	
	static Map config;

	static SequenceDataset hmmDb = new SequenceDataset();
	static PredictionDataset pd;

	static List<Integer> KList;
	static int maxIterHmm;
	static int maxIterEhmm;
	static int maxIterShareHmm;
	static boolean avgTest;
	static List<Integer> numStateList; // the list of numbers of states for HMM.
	static List<Integer> numComponentList; // the list of numbers of GMM components for HMM.
	static List<Double> sparsityList; // the list of values of sparsity's coefficients.
	static int numStateHmm; // the default number of states for HMM.
	static int numComponentHmm; // the default number of GMM components for HMM
	static int numClusterEhmm; //the default number of ehmm for user grouping
	static int numStateShareHmm; // the default number of states for ShareHMM.
	static int numComponentShareHmm; // the default number of components for ShareHMM
	static double sparsity; // the default value of sparsity for HMM
	static List<String> initMethodList; // the list of initalization methods for EHMM.
	static boolean evalNumState; // whether to evaluate numState
	static boolean evalNumComponent; // whether to evaluate numComponent
	static boolean evalSparsity; // whether to evaluate sparsity

	// parameters for prediction
	static double distThre;
	static double timeThre;
	static boolean filterTest;
	static boolean evalParas;

	// parameters for experiments
	public static String underlyingDistribution;
	public static String dataset;
	public static String method;
	public static boolean printLL;
	public static Boolean printDetail;
	public static Boolean printEstimator;
	public static Boolean loadModel;
	public static Boolean saveModel;
	public static Boolean saveUserAcc;
	public static String filename;
	public static String header;
	public static long programStart;
	public static long programEnd;
	public static double runtime;
	public static List<Double> accList;
	
	// results for users
	public static List<Map<String, List<Double>>> userAccList;
	/*** ---------------------------------- Initialize ---------------------------------- ***/
	static void init(String paraFile, int number) throws Exception {
		config = new Config().load(paraFile);
		loadData(number);
		if(option == "default")
		{
			KList = new ArrayList<Integer>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64));
		}	
	}
	/*** ---------------------------------- Input and Output ------------------------------- ***/
	static void loadData(int number) throws Exception {
		// load data
		String sequenceFile = (String) ((Map) ((Map) config.get("file")).get("input")).get("sequences");//find dataset sequences
		System.out.println(sequenceFile);
		dataset = (String) ((Map) config.get("exp")).get("dataset");//establish the dataset path
		underlyingDistribution = (String) ((Map) config.get("exp")).get("underlyingDistribution");//decide underlyingDistribution
		System.out.println(underlyingDistribution);
		double testRatio = (Double) ((Map) config.get("predict")).get("testRatio");//decide the training and test dataset ratio
		filterTest = (Boolean) ((Map) config.get("predict")).get("filterTest");//set the filterTest yes
	
		hmmDb = null;//initialize the dataset

		hmmDb = new SequenceDataset();
		
		
		hmmDb.load(sequenceFile + number + ".txt", testRatio, dataset, underlyingDistribution, filterTest);

		
		pd = null;//initialize the test dataset
		
		pd = hmmDb.extractTestData();                          
				
		distThre = (Double) ((Map) config.get("predict")).get("distThre");// distance threshold for forming the candidate pool
		timeThre = (Double) ((Map) config.get("predict")).get("timeThre");//time threshold
		pd.genCandidates(distThre, timeThre);

		// the hmm parameters
		maxIterHmm = (Integer) ((Map) config.get("hmm")).get("maxIter");
		KList = (List<Integer>) ((Map) config.get("predict")).get("K");//k和component是什么意思,查代码是做EM算法的两个参数，k=userNumCorrectK
		avgTest = (Boolean) ((Map) config.get("predict")).get("avgTest");
		initMethodList = (List<String>) ((Map) config.get("ehmm")).get("initMethod");
		
		numStateHmm = (Integer) ((Map) config.get("hmm")).get("numState");
		numComponentHmm = (Integer) ((Map) config.get("hmm")).get("numComponent");
		
		// the ehmm parameters
		maxIterEhmm = (Integer) ((Map) config.get("ehmm")).get("maxIter");
		numClusterEhmm = (Integer) ((Map) config.get("ehmm")).get("numCluster");
		
		// the sharehmm parameters
		maxIterShareHmm = (Integer) ((Map) config.get("sharehmm")).get("maxIter");
		sparsity = (Double) ((Map) config.get("sharehmm")).get("sparsity");
		numStateShareHmm = (Integer) ((Map) config.get("sharehmm")).get("numState");
		numComponentShareHmm = (Integer) ((Map) config.get("sharehmm")).get("numComponent");
		
		// list of parameters for evaluation
		numStateList = (List<Integer>) ((Map) config.get("sharehmm")).get("numStateList");
		numComponentList = (List<Integer>) ((Map) config.get("sharehmm")).get("numComponentList");
		sparsityList = (List<Double>) ((Map) config.get("sharehmm")).get("sparsityList");
		evalNumState = (Boolean) ((Map) config.get("sharehmm")).get("evalNumState");
		evalNumComponent = (Boolean) ((Map) config.get("sharehmm")).get("evalNumComponent");
		evalSparsity = (Boolean) ((Map) config.get("sharehmm")).get("evalSparsity");
		
		// whether to print or save
		printLL = (Boolean) ((Map) config.get("exp")).get("printLL");
		printDetail = (Boolean) ((Map) config.get("exp")).get("printDetail");
		printEstimator = (Boolean) ((Map) config.get("exp")).get("printEstimator");
		loadModel = (Boolean) ((Map) config.get("exp")).get("loadModel");
		saveModel = (Boolean) ((Map) config.get("exp")).get("saveModel");
		saveUserAcc = (Boolean) ((Map) config.get("exp")).get("saveUserAcc");
	}
	//save the result to file
	static void writeIntoFile(String header, List<Double> accList) {
		BufferedWriter out = null;
		try {
			out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename, true)));
			out.write(header);
			for (double acc : accList)
				out.write(acc + " ");
			out.write("\n");
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				out.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	/*** ---------------------------------- Train and Predict ---------------------------------- ***/
	static void runCmp(int counter) throws Exception {
		evalParas = false;
		filename = "../result/txt/acc_" + dataset + ".txt";
		
		if(underlyingDistribution.equals("2dGaussian"))
			runDistance(counter);
			

		runSparse(maxIterShareHmm, numStateShareHmm, numComponentShareHmm, sparsity, counter);
	}
	static void beginExp(){
		programStart = System.currentTimeMillis();
		System.out.println("Start "+ header + "training.");
	}
	static void endExp(){
		System.out.println(header + "based prediction accuracy: " + accList);
		programEnd = System.currentTimeMillis();
		runtime = (programEnd - programStart) / 1000.0;
		System.out.println(header + "takes " + runtime / 60.0 + "mins.");
		header = header + runtime + " ";
		writeIntoFile(header, accList);
		


	}
	/*** Run models with fine-tuned parameters 
	 * @throws IOException ***/
	static void runDistance( int counter) throws IOException {
		header = "Law ";
		method = "Law";
		beginExp();	
		
		//set the file directory
		utils.genreateOutputFiles( header+"Distance" +"\n", SAVE_PATH);   
		
		//Prediction calculation of test set PD
		DistancePredictor dp = new DistancePredictor();
		dp.predict(pd, KList,counter);
		accList = dp.getAccuracy();
		
        dp.printAccuracy();
    
		endExp();
	}
	
	static void runHMM(int maxIter, int numStates, int numComponent,int counter) throws Exception {
		header = "HMM ";
		method = "HMM";
		beginExp();
		utils.genreateOutputFiles( header + counter +"\n", SAVE_PATH);  		
		
		HMM h = new HMM(maxIter, underlyingDistribution);
		h.train(hmmDb, numStates, numComponent);

       // System.out.println("model" + h.calcGeoLL(null));
		
		HMMPredictor hp = new HMMPredictor(h, avgTest);

		hp.predict(pd, KList,counter);
		accList = hp.getAccuracy();
   		
		endExp();
	}

	static void runGmove(int maxIter, int numCluster, int numStates, int numComponent, String initMethod,int counter) throws Exception {
		header = "Gmove ";
		method = "Gmove";
		beginExp();
		utils.genreateOutputFiles( header+ "\n", SAVE_PATH);  		
		
		EHMM ehmm = new EHMM(maxIter, numStates, numStates, numComponent, numCluster, initMethod, underlyingDistribution);
		ehmm.train(hmmDb);
		EHMMPredictor ep = new EHMMPredictor(ehmm, avgTest);
		ep.predict(pd, KList,counter);
		accList = ep.getAccuracy();
 		
		endExp();
	}
	static void runPipeline(int maxIter, int numStates, int numComponent,int counter) throws Exception {
		header = "Pipeline ";
		method = "Pipeline";
		beginExp();
		utils.genreateOutputFiles( header+ "\n", SAVE_PATH);  				
		
		Pipeline pipeline = new Pipeline(maxIter, underlyingDistribution);
		pipeline.train(hmmDb, numStates, numComponent);
		ShareHMMPredictor hp = new ShareHMMPredictor(pipeline, avgTest);
		hp.predict(pd, KList,counter);
		accList=hp.getAccuracy();
		endExp();
	}
	static void runHard(int maxIter, int numStates, int numComponent,int counter) throws Exception {
		header = "Hard ";
		method = "Hard";
		utils.genreateOutputFiles( header+ "\n", SAVE_PATH);  				
		
		runSparse(maxIter, numStates, numComponent, Double.POSITIVE_INFINITY,counter);
	}
	
	static void runSoft(int maxIter, int numStates, int numComponent,int counter) throws Exception {
		header = "Soft ";
		method = "Soft";
		utils.genreateOutputFiles( header + counter + "\n", SAVE_PATH);  		
		runSparse(maxIter, numStates, numComponent, 0, counter);
	}

	static void runSparse(int maxIter, int numStates, int numComponent, double sparsity, int counter) throws Exception {
		if(sparsity != 0 && sparsity != Double.POSITIVE_INFINITY && !evalParas){
			header = "Sparse ";
			method = "Sparse";
		}
		beginExp();
		utils.genreateOutputFiles( header + "\n", SAVE_PATH);  				
		
		ShareHMM shareHMM = new ShareHMM(maxIter, underlyingDistribution);
		
		shareHMM.train(hmmDb, numStates, numComponent, sparsity, underlyingDistribution);	
		if(saveModel) shareHMM.saveModel();
		ShareHMMPredictor hp = new ShareHMMPredictor(shareHMM, avgTest);
		
		
		
		double time_start = System.currentTimeMillis();
		
		hp.predict(pd, KList, counter);
		
		double time_end  = System.currentTimeMillis();
		double t = time_end - time_start;     

		if(city == "xian")
				System.out.println("xian time is " + t*251/ pd.size() + "ms"); 
		else if(city == "chengdu") 
			System.out.println("chengdu time is " + t*505/ pd.size() + "ms"); 
		else if(city=="beijing")
			System.out.println("chengdu time is " + t*64/ pd.size() + "ms"); 
		accList=hp.getAccuracy();
		endExp();

		utils.genreateOutputFiles("Sparse  " + counter +"  based prediction accuracy: " + accList + "\n", ACC_SAVE_PATH);
		
	}
	
	/*** ---------------------------------- Evaluate different parameters. ---------------------------------- ***/
	static void runEval(int counter) throws Exception{
		evalParas = true;
		if(evalNumState) evalNumStates(counter);
		if(evalNumComponent) evalNumComponents(counter);
		if(evalSparsity) evalSparsity(counter);
	}
	
	static void evalNumStates(int counter) throws Exception {
		System.out.println("Evaluation on NumStates...");
		filename = "../result/txt/evalNumStates_" + dataset + ".txt";
		for (Integer numState : numStateList) {
			System.out.println("maxIter:" + maxIterShareHmm + " " + "numState:" + numState + " " + "numComponent:" + numComponentShareHmm + " " + "sparsity:" + sparsity);
			header = numState + " "; runSparse(maxIterShareHmm, numState, numComponentShareHmm, sparsity,counter);
		}
	}
	static void evalNumComponents(int counter) throws Exception {
		System.out.println("Evaluation on NumComponents...");
		filename = "../result/txt/evalNumComponents_" + dataset + ".txt";
		for (Integer numComponent : numComponentList) {
			System.out.println("maxIter:" + maxIterShareHmm + " " + "numState:" + numStateShareHmm + " " + "numComponent:" + numComponent + " " + "sparsity:" + sparsity);
			header = numComponent + " "; runSparse(maxIterShareHmm, numStateShareHmm, numComponent, sparsity,counter);
		}
	}
	static void evalSparsity(int counter) throws Exception {
		System.out.println("Evaluation on Sparsity...");
		filename = "../result/txt/evalSparsity_" + dataset + ".txt";
		for (Double sparsity : sparsityList) {
			System.out.println("maxIter:" + maxIterShareHmm + " " + "numState:" + numStateShareHmm + " " + "numComponent:" + numComponentShareHmm + " " + "sparsity:" + sparsity);
			header = sparsity + " "; runSparse(maxIterShareHmm, numStateShareHmm, numComponentShareHmm, sparsity,counter);
		}
	}
	
	/*** ---------------------------------- Main ---------------------------------- ***/
	public static void main(String[] args) throws Exception {
		if(city == "beijing")
			for(int counter=0; counter<=1207; counter++)
			{		
				String paraFile = "../scripts/parameter File.yaml"; // reading the parameter File
				System.out.println("counter is " + counter);
				init(paraFile, counter);
				runCmp(counter);
				hmmDb = null;
				pd=null;
			}

	}
}