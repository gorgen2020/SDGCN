package model;

import java.util.*;

import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import com.mongodb.gridfs.GridFS;
import myutils.*;
import org.bson.BSON;
import predict.EHMMPredictor;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import cluster.KMeans;
import data.Checkin;
import data.CheckinDataset;
import data.PredictionDataset;
import data.Sequence;
import data.SequenceDataset;
import distribution.Multinomial;

// Ensemble of HMMs
public class EHMM {
	int C; // The number of clusters (every cluster is corresponding to a hmm).
	int MaxIter;
	int BG_numState;
	int HMM_K;
	int HMM_M;
	String initMethod;
	String underlyingDistribution;
	double elapsedTime;

	ArrayList<HMM> hmms;
	double[][] seqsFracCounts;
	public HashMap<Long, HashSet<Integer>> user2seqs = new HashMap<Long, HashSet<Integer>>();
	double totalLL = 0;
	SequenceDataset data;

	public EHMM(int MaxIter, int BG_numState, int HMM_K, int HMM_M, int C, String initMethod, String underlyingDistribution) {
		this.MaxIter = MaxIter;
		this.BG_numState = BG_numState;
		this.HMM_K = HMM_K;
		this.HMM_M = HMM_M;
		this.C = C;
		this.initMethod = initMethod;
		this.underlyingDistribution = underlyingDistribution;
		hmms = new ArrayList<HMM>(C);
	}
	
//	public EHMM(DBObject o, SequenceDataset data) {
//		this.data = data;
//		calcUser2seqs();
//		load(o);
//	}

	public void train(SequenceDataset data) throws Exception {
		long start = System.currentTimeMillis();
		this.data = data;
		seqsFracCounts = new double[C][data.size()];
		calcUser2seqs();
		initHMMs();
		double prevLL = totalLL;
		for (int iter = 0; iter < MaxIter; iter++) {
			calcTotalLL();
			System.out.println("EHMM finished iteration " + iter + ". Log-likelihood:" + totalLL);

			eStep();
			mStep();

			if (Math.abs(totalLL - prevLL) <= 0.01)
				break;
			prevLL = totalLL;
		}
		long end = System.currentTimeMillis();
		elapsedTime = (end - start) / 1000.0;
	}

//	public void testWhileTrain(SequenceDataset data, boolean avgTest) throws Exception {
//		PredictionDataset pd = data.extractTestData();
//		pd.genCandidates(3, 240);
//
//		this.data = data;
//		seqsFracCounts = new double[C][data.size()];
//		calcUser2seqs();
//		initHMMs();
//		double prevLL = totalLL;
//		for (int iter = 0; iter < MaxIter; iter++) {
//			calcTotalLL();
//			System.out.println("EHMM finished iteration " + iter + ". Log-likelihood:" + totalLL);
//
//			EHMMPredictor up = new EHMMPredictor(this, avgTest);
//			up.predict(pd, 3);
//			up.printAccuracy();
//
//			eStep();
//			mStep();
//
//			if (Math.abs(totalLL - prevLL) <= 0.01)
//				break;
//			prevLL = totalLL;
//		}
//		EHMMPredictor up = new EHMMPredictor(this, avgTest);
//		up.predict(pd, 3);
//		up.printAccuracy();
//	}

	private void calcTotalLL() {
		totalLL = 0;
		for (HMM hmm : hmms) {
			totalLL += hmm.getTotalLL();
		}
	}

	public void calcUser2seqs() {
		for (int i = 0; i < data.size(); i++) {
			Sequence seq = data.getSequence(i);
			long user = seq.getUserId();
			if (!user2seqs.containsKey(user)) {
				user2seqs.put(user, new HashSet<Integer>());
			}
			user2seqs.get(user).add(i);
		}
//		int num = 0;
//		for (Long i : user2seqs.keySet()){
//			System.out.println(user2seqs.get(i).size());
//			if(user2seqs.get(i).size()>=10)
//				num++;
//		}
//		System.out.println("user number with no less than 10 trajs:" + num);
		System.out.println("user number:" + user2seqs.size());
	}

	private void initHMMs() {
		if (initMethod.equals("random")) {
			SplitDataRandomly();
		}
		if (initMethod.equals("uniform")) {
			SplitDataUniformly();
		}
		if (initMethod.equals("kmeans_k")) {
			SplitDataByKMeans(false);
		}
		if (initMethod.equals("kmeans_2k")) {
			SplitDataByKMeans(true);
		}
		for (int c = 0; c < C; ++c) {
			HMM hmm = new HMM(MaxIter, underlyingDistribution);
			hmm.train(data, HMM_K, HMM_M, seqsFracCounts[c]);
			hmms.add(hmm);
		}
	}

	private void SplitDataUniformly() {
		for (int i = 0; i < data.size(); i++) {
			for (int c = 0; c < C; ++c) {
				seqsFracCounts[c][i] = 1.0 / C;
			}
		}
	}

	private void SplitDataRandomly() {
		Random random = new Random(1);
		for (long user : user2seqs.keySet()) {
			double[] seqFracCounts = new double[C];
			for (int c = 0; c < C; ++c) {
				seqFracCounts[c] = random.nextDouble();
			}
			ArrayUtils.normalize(seqFracCounts);
			for (int i : user2seqs.get(user)) {
				for (int c = 0; c < C; ++c) {
					seqsFracCounts[c][i] = seqFracCounts[c];
				}
			}
		}
	}

	private void SplitDataByKMeans(boolean useTwiceLongFeatures) {
		CheckinDataset bgd = new CheckinDataset();
		bgd.load(data);
		Background b = new Background(MaxIter);
		b.train(bgd, BG_numState);
		List<RealVector> featureVecs = new ArrayList<RealVector>();
		List<Double> weights = new ArrayList<Double>();
		HashMap<Integer, Long> u2user = new HashMap<Integer, Long>(); // u is the index of user
		int u = 0;
		for (long user : user2seqs.keySet()) {
			RealVector featureVec;
			if (useTwiceLongFeatures) { // use BG_numState*2 dimension feature vectors
				featureVec = new ArrayRealVector(BG_numState * 2);
				for (int state = 0; state < BG_numState; ++state) {
					for (int n = 0; n < 2; ++n) {
						for (int i : user2seqs.get(user)) {
							double membership = 0;
							if(underlyingDistribution.equals("2dGaussian")){
								membership = b.calcLL(data.getGeoDatum(2 * i + n));
							}
							else if(underlyingDistribution.equals("multinomial")){
								membership = b.calcLL(data.getItemDatum(2 * i + n));
							}
//							double membership = b.calcLL(data.getGeoDatum(2 * i + n));
							featureVec.addToEntry(2 * state + n, membership / user2seqs.get(user).size());
						}
						//						featureVec.setEntry(2 * state + n, Math.exp(featureVec.getEntry(2 * state + n))); // transform to probability
					}
				}
			} else { // use BG_numState dimension feature vectors
				featureVec = new ArrayRealVector(BG_numState);
				for (int state = 0; state < BG_numState; ++state) {
					for (int i : user2seqs.get(user)) {
						double membership = 0;
						if(underlyingDistribution.equals("2dGaussian")){
							membership = b.calcLL(data.getGeoDatum(2 * i), data.getGeoDatum(2 * i + 1));
						}
						else if(underlyingDistribution.equals("multinomial")){
							membership = b.calcLL(data.getItemDatum(2 * i), data.getItemDatum(2 * i + 1));
						}
//						double membership = b.calcLL(data.getGeoDatum(2 * i), data.getGeoDatum(2 * i + 1));
						featureVec.addToEntry(state, membership / user2seqs.get(user).size());
					}
					//					featureVec.setEntry(state, Math.exp(featureVec.getEntry(state))); // transform to probability
				}
			}
			featureVecs.add(u, featureVec);
			weights.add(u, 1.0);
			u2user.put(u, user);
			++u;
		}
		KMeans kMeans = new KMeans(500);
		List<Integer>[] kMeansResults = kMeans.cluster(featureVecs, weights, C);
//		System.out.println(featureVecs.size());
//		for (List<Integer> kMeansResult : kMeansResults) {
//			System.out.println(kMeansResult.size());
//		}
		for (int c = 0; c < C; ++c) {
			List<Integer> members = kMeansResults[c];
			for (int member : members) {
				long user = u2user.get(member);
				for (int i : user2seqs.get(user)) {
					seqsFracCounts[c][i] = 1;
					for (int other_c = 0; other_c < C; ++other_c) {
						if (other_c != c) {
							seqsFracCounts[other_c][i] = 0;
						}
					}
				}
			}
		}
	}

	private void eStep() {
		for (long user : user2seqs.keySet()) {
			double[] posteriors = getPosteriors(user);
			for (int i : user2seqs.get(user)) {
				for (int c = 0; c < C; ++c) {
					seqsFracCounts[c][i] = posteriors[c];
				}
			}
		}
	}

	private void mStep() {
		for (int c = 0; c < C; ++c) {
			HMM hmm = hmms.get(c);
			hmm.update(data, seqsFracCounts[c]);
		}
	}

	private double[] getPosteriors(long user) {
		double[] posteriors = new double[C];
		if (user2seqs.containsKey(user)) {
			for (int c = 0; c < C; ++c) {
				for (int i : user2seqs.get(user)) {
					Sequence seq = data.getSequence(i);
					posteriors[c] += hmms.get(c).calcSeqScore(seq);
				}
			}
		}
		ArrayUtils.logNormalize(posteriors);
		return posteriors;
	}

	public double calcGeoLL(long user, List<RealVector> geo, boolean avgTest) {
		double LL = 0;
		double[] posteriors = getPosteriors(user); // The posteriors here serve as priors.
		for (int c = 0; c < C; ++c) {
			LL += posteriors[c] * hmms.get(c).calcGeoLL(geo, avgTest);
		}
		return LL;
	}

	public double calcItemLL(long user, List<Integer> items, boolean avgTest) {
		double LL = 0;
		double[] posteriors = getPosteriors(user); // The posteriors here serve as priors.
		for (int c = 0; c < C; ++c) {
			LL += posteriors[c] * hmms.get(c).calcItemLL(items, avgTest);
		}
		return LL;
	}
	
	/**
	 * To-Do: convert to BSon and load
	 */

//	public DBObject statsToBson() {
//		DBObject o = new BasicDBObject();
//		o.put("C", C);
//		o.put("K", HMM_K);
//		o.put("Init", initMethod);
//		o.put("time", elapsedTime);
//		return o;
//	}
//	
//	public DBObject toBson() {
//		DBObject o = new BasicDBObject();
//		o.put("C", C);
//		o.put("MaxIter", MaxIter);
//		o.put("BG_numState", BG_numState);
//		o.put("K", HMM_K);
//		o.put("M", HMM_M);
//		o.put("Init", initMethod);
//		List<DBObject> dbHmms = new ArrayList<DBObject>();
//		for (HMM hmm : this.hmms)
//			dbHmms.add(hmm.toBson());
//		o.put("hmms", dbHmms);
//		return o;
//	}
//	
//	public void load(DBObject o) {
//		this.C = (Integer) o.get("C");
//		this.MaxIter = (Integer) o.get("MaxIter");
//		this.BG_numState = (Integer) o.get("BG_numState");
//		this.HMM_K = (Integer) o.get("K");
//		this.HMM_M = (Integer) o.get("M");
//		this.initMethod = (String) o.get("Init");
//		List<DBObject> dbHmms = (List<DBObject>) o.get("hmms");
//		this.hmms = new ArrayList<HMM>(C);
//		for(DBObject dbHmm:dbHmms){
//			this.hmms.add(new HMM(dbHmm));
//		}
//	}
}
