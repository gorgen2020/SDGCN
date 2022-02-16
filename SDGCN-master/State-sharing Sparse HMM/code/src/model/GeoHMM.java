package model;

import com.mongodb.DBObject;
import data.SequenceDataset;
import org.apache.commons.math3.linear.RealVector;

import java.util.List;
import java.util.Map;

/**
 * Created by chao on 1/11/16.
 * This class is the HMM that uses only geographical information.
 */
public class GeoHMM extends HMM {

    public GeoHMM(int maxIter) {
        super(maxIter);
    }


    /**
     * Step 2.2: learning the parameters using EM: M-Step.
     */
    protected void mStep(SequenceDataset data) {
        updatePi();
        updateA();
        updateGeoModel(data);
//        updateTemporalModel(data);
    }

    protected double calcLLState(RealVector geoDatum, RealVector temporalDatum, Map<Integer, Integer> textDatum, int k,
                                 boolean isTest) {
        double geoProb = calcGeoLLState(geoDatum, k);
//        double temporalProb = temporalModel[k].calcLL(temporalDatum);
//        return geoProb + temporalProb;
        return geoProb;
    }

}
