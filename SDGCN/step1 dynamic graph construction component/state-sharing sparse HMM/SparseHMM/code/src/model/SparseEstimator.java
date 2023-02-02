package model;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

public class SparseEstimator {
	public static Boolean toOutput = false;
//	public static Boolean toSave = true;
	public String method = "Newton"; // "Newton" or "Dichotomy"
	public String methodCalMean = "harmonic"; // "arithmetic", "logarithmic" or "harmonic"
	

	
	public double lambda; // The coefficient of the sparse constraint item: lambda * \sum{- B * log(B)}
	public RealVector sumGamma = null;
	public RealVector alpha = null;
	public double objective;
	RealVector alphaPrev = null;
	double eta; // Lagrangian coefficient

	int dimension;
	
	double innerLoopThres = 1e-3;// least: 1e-15
	double outerLoopThres = 1e-3;// least: 1e-15
	
//	List<Double> innerLoopErrorList; // the list of errors in the inner loop. (Newton's method)
	List<Double> outerLoopErrorList = new ArrayList<Double>();; // the list of errors in the outer loop. (CCCP)

	public SparseEstimator() {
	}

	public SparseEstimator(double lambda) {
		this.lambda = lambda;
	}

	public RealVector estimate(RealVector sumGamma) {
		this.dimension = sumGamma.getDimension();
		double sumSumGamma = sumGamma.getL1Norm();
		if(sumSumGamma == 0)
			System.out.println("sumSumGamma is 0. Error.");
		this.sumGamma = sumGamma.mapDivide(sumSumGamma);
		lambda = lambda / sumSumGamma;
		if (lambda == 0 || Math.abs(lambda / sumGamma.getL1Norm()) < 0.001) { // no sparsity
			if (toOutput)
				System.out.println("lambda is " + lambda + ". No sparsity.");
			alpha = sumGamma.mapDivide(sumGamma.getL1Norm());
		} else if (lambda == Double.POSITIVE_INFINITY || Math.abs(lambda / sumGamma.getL1Norm()) > 1000) {
			// the most sparse (i.e. hard assignment)
			if (toOutput)
				System.out.println("lambda is " + lambda + ". Hard assignment.");
			alpha = new ArrayRealVector(sumGamma.getDimension());
			alpha.setEntry(sumGamma.getMaxIndex(), 1.0);
		} else { // the ordinary cases
			if (toOutput)
				System.out.println("lambda is " + lambda + ".");
			CCCP(); // direct operation on this.alpha
		}
		if (toOutput)
			System.out.println("alpha is estimated as " + alpha + ".");
		return alpha;
	}

	void CCCP() {
		if (toOutput)
			System.out.println("Begin to run CCCP...");
		alpha = sumGamma.mapDivide(sumGamma.getL1Norm());
		int iter = 0;
		double diff = 1;
		
		while (diff > outerLoopThres) {
			calculateObjective();
			alphaPrev = alpha.copy();
			solveEta();
			updateAlpha();
			diff = alpha.getL1Distance(alphaPrev);
			outerLoopErrorList.add(diff);
			if (toOutput){
				System.out.println("CCCP finished iteration " + iter + ". Diff: " + diff + ".");
				System.out.println("Alpha is estimated as " + alpha + ".");
			}
			iter++;
		}
	}

	void calculateObjective() {
		objective = 0;
//		for (int i = 0; i < sumGamma.getDimension(); i++) {
//			if (sumGamma.getEntry(i) > 0) {
//				objective += -sumGamma.getEntry(i) * Math.log(alpha.getEntry(i))
//						- lambda * alpha.getEntry(i) * Math.log(alpha.getEntry(i));
//			}
//		}
		for (int i = 0; i < sumGamma.getDimension(); i++) {
			if (sumGamma.getEntry(i) > 0) {
				objective += -sumGamma.getEntry(i) * Math.log(alpha.getEntry(i))
						- lambda * alpha.getEntry(i) * Math.log(alpha.getEntry(i));
			}
		}
	}

	void solveEta() {
		switch (method) {
		case "Newton": runNewton(); break;
		case "Dichotomy": runDichotomy(); break;
		default:
			if (toOutput)
				System.out.println("Invalid method name. Please use one of the below:\nNewton or Dichotomy");
		}
	}

	void runNewton() {
		if (toOutput)
			System.out.println("  Begin to run Newton's method...");
		eta = calEtaLower(); // guarantee convergence
		double derivative;
		double sum = 0;
		double residual = 1;
		int iter = 0;
		while (Math.abs(residual) > innerLoopThres) {
			sum = calSum();
			residual = sum - 1;
			// sum monotonically decreases with eta
			derivative = calDer();
			eta += residual / derivative;
			if (toOutput)
				System.out.println("  Newton's method finished iteration " + iter + ". Residual: " + residual + ". Eta: "
						+ eta + ".");
			iter++;
			if (residual == -1)
				break;
		}
	}

	void runDichotomy() {
		if (toOutput)
			System.out.println("  Begin to run dichotomy method...");
		double etaLower = calEtaLower();
		double etaUpper = calEtaUpper();
		
		switch (methodCalMean) {
		case "arithmetic": eta = (etaLower + etaUpper) / 2; break;
		case "logarithmic": eta = Math.sqrt(etaLower * etaUpper); break;
		case "harmonic": eta = 2 / (1 / etaLower + 1 / etaUpper); break;
		default:
			if (toOutput)
				System.out.println("  Invalid methodCalMean name. Please use one of the below:\narithmetic, logarithmic or harmonic");
		}
		double sum = 0;
		double residual = 1;
		int iter = 0;
		while (Math.abs(residual) > innerLoopThres) {
			sum = calSum();
			residual = sum - 1;
			// sum monotonically decreases with eta
			if (residual > 0)
				etaLower = eta;
			else
				etaUpper = eta;
			eta = (etaLower + etaUpper) / 2;
			if (toOutput)
				System.out.println("  Dichotomy method finished iteration " + iter + ". Residual: " + residual + ". "
						+ "Lower bound: " + etaLower + ". Upper bound: " + etaUpper);
			iter++;
			if (residual == -1)
				break;
		}
	}
	
	double calEtaLower(){
		int maxAlphaIdx = alpha.getMaxIndex();
		double etaLower = sumGamma.getEntry(maxAlphaIdx) / lambda + Math.log(alpha.getEntry(maxAlphaIdx)) + 1;
		return etaLower;
	}
	
	double calEtaUpper(){
		double maxAlphaValue = alpha.getMaxValue();
		double etaUpper = 1 / lambda + Math.log(maxAlphaValue) + 1;
		return etaUpper;
	}

	double calSum() {
		double sum = 0;
		for (int i = 0; i < sumGamma.getDimension(); i++) {
			if (alpha.getEntry(i) > 1e-6) {
				sum += sumGamma.getEntry(i) / lambda / (eta - 1 - Math.log(alpha.getEntry(i)));
			}
		}
		return sum;
	}

	double calDer() {
		double sum = 0;
		double temp;
		for (int i = 0; i < sumGamma.getDimension(); i++) {
			if (alpha.getEntry(i) > 1e-6) {
				temp = eta - 1 - Math.log(alpha.getEntry(i));
				sum += sumGamma.getEntry(i) / lambda / (temp * temp);
			}
		}
		return sum;
	}

	void updateAlpha() {
		for (int i = 0; i < sumGamma.getDimension(); i++) {
			if (alpha.getEntry(i) > 0) { // 1e-6
				alpha.setEntry(i, sumGamma.getEntry(i) / lambda / (eta - 1 - Math.log(alpha.getEntry(i))));
			} else {
				alpha.setEntry(i, 0);
			}
		}
	}

	RealVector getAlpha() {
		return alpha;
	}
	
	static void testConvergence() throws IOException {
		RealVector sumGamma = new ArrayRealVector(new double[] { 0.0, 0.1, 0.2, 0.3, 0.4});
		SparseEstimator sparseEstimator = new SparseEstimator(1.0);
		RealVector alpha = sparseEstimator.estimate(sumGamma);		
		// write the results into file
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("../result/txt/synthetic_convergence.txt", false)));
		for (double diff: sparseEstimator.outerLoopErrorList)
			out.write(diff + " ");
		out.write("\n");
		out.close();
	}
	
	static void testEffectiveness() throws IOException {
		toOutput = true;
		RealVector sumGamma = new ArrayRealVector(new double[] { 0.0, 0.1, 0.2, 0.3, 0.4});
		// write the results into file
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("../result/txt/synthetic_effectiveness.txt", false)));
		for (int i = -20; i <=20 ;i++){
			double lambda = Math.pow(10.0, i/10.0);
			SparseEstimator sparseEstimator = new SparseEstimator(lambda);
			RealVector alpha = sparseEstimator.estimate(sumGamma);
			out.write(String.valueOf(sparseEstimator.lambda));
			for(int j = 0; j < sparseEstimator.dimension; j++)
				out.write(" " + sparseEstimator.alpha.getEntry(j));
			out.write("\n");
		}
		out.close();
	}

	/** ---------- Main ---------- **/
	public static void main(String[] args) throws Exception {
		testConvergence();
//		testEffectiveness();
	}
}
