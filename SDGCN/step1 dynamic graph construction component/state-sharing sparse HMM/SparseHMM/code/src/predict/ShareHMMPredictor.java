package predict;

import data.Checkin;
import data.PredictionDataset;
import data.Sequence;
import data.SequenceDataset;
import demo.Demo;
import model.ShareHMM;
import myutils.ScoreCell;
import myutils.TopKSearcher;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ShareHMMPredictor extends Predictor {

	ShareHMM model;
	boolean avgTest;

	public ShareHMMPredictor(ShareHMM model, boolean avgTest) {
		this.model = model;
		this.avgTest = avgTest;
	}

	public ScoreCell calcScore(Sequence m, Checkin p) {
		Checkin startPlace = m.getCheckin(0);
		long userId = m.getCheckin(0).getUserId();
		int u = SequenceDataset.userIdList.indexOf((long)userId);
//		int u = model.userList.indexOf(userid);
		double score = 0;
		int checkinId = -1;

		
		if(Demo.underlyingDistribution.equals("2dGaussian")){
			List<RealVector> geo = new ArrayList<RealVector>();
			geo.add(startPlace.getLocation().toRealVector());
			geo.add(p.getLocation().toRealVector());
			score = model.calcGeoLL(geo, u);
			checkinId = p.getId();
		}
		else if(Demo.underlyingDistribution.equals("multinomial")){
			List<Integer> items = new ArrayList<Integer>();
			items.add(startPlace.getItemId());
			items.add(p.getItemId());
			

			score = model.calcItemLL(items, u);

			
			checkinId = p.getItemId(); // do not distinguish checkin, i.e., as long as the item id is right, the prediction is correct.
		}
		
		
		return new ScoreCell(checkinId, score);
	}
	
//	public void predictItem(PredictionDataset mdb, List<Integer> KList) {
//		int[] numCorrectK = new int[KList.size()];
//        int Kmax = KList.get(KList.size()-1);
//        
//        for (int i=0; i<mdb.size(); i++) {
//        	System.out.println("It is predicting for " + i + "-th seq.");
//            Sequence m = mdb.getSeq(i);
//            int[] nextItemTopK = model.nextItemTopK(m.getCheckin(0).getItemId(), (int) m.getCheckin(0).getUserId(), Kmax);
//            for (int j = 0; j < KList.size(); j++) {
//            	for (int k = 0; k < KList.get(j); k++){
//            		// result get Id is equal to checkin item Id
//	            	if (m.getCheckin(1).getItemId() == nextItemTopK[k]) {
//	            		numCorrectK[j]++;
//	                } else {
//	                }
//            	}
//            }
//        }
//        for (int j = 0; j < KList.size(); j++){
//        	accuracyList.add((double) numCorrectK[j] / (double) mdb.size());
//        }
//	}

	public void printAccuracy() {
		System.out.println("HMM-based predictor accuracy:" + accuracyList);
	}

}
