package predict;

import data.Checkin;
import data.PredictionDataset;
import data.Sequence;
import demo.Demo;
import model.HMM;
import model.EHMM;
import myutils.ScoreCell;
import myutils.TopKSearcher;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class EHMMPredictor extends Predictor {

	EHMM model;
	boolean avgTest;

	public EHMMPredictor(EHMM model, boolean avgTest) {
		this.model = model;
		this.avgTest = avgTest;
	}

	public ScoreCell calcScore(Sequence m, Checkin p) {
		Checkin startPlace = m.getCheckin(0);
		double score = 0;
		int checkinId = -1;
		if(Demo.underlyingDistribution.equals("2dGaussian")){
			List<RealVector> geo = new ArrayList<RealVector>();
			geo.add(startPlace.getLocation().toRealVector());
			geo.add(p.getLocation().toRealVector());
			score = model.calcGeoLL(m.getUserId(), geo, avgTest);
			checkinId = p.getId();
		}
		else if(Demo.underlyingDistribution.equals("multinomial")){
			List<Integer> items = new ArrayList<Integer>();
			items.add(startPlace.getItemId());
			items.add(p.getItemId());
			score = model.calcItemLL(m.getUserId(), items, avgTest);
			checkinId = p.getItemId();
		}
		return new ScoreCell(checkinId, score);
	}

	public void printAccuracy() {
		System.out.println("EHMM-based predictor accuracy:" + accuracyList);
	}

}
