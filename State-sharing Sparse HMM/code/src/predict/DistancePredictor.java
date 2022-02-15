package predict;

import data.Checkin;
import data.Sequence;
import myutils.ScoreCell;

/**
 * Created by chao on 5/3/15.
 */
public class DistancePredictor extends Predictor {
	private int i=20;
    public ScoreCell calcScore(Sequence m, Checkin p) {
        int placeId = p.getId();
        Checkin startPlace = m.getCheckin(0);
        double score = p.getLocation().calcEuclideanDist(startPlace.getLocation());
        

       if (i>0)
       {
            System.out.println("º∆À„µ√µΩ"+placeId+ "score"+score ); 
       		i--;
       }
        return new ScoreCell(placeId, score);
    }

    public void printAccuracy() {
        System.out.println("Distance-based predictor accuracy:" + accuracyList);
    }

}
