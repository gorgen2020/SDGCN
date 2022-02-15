package model;

import cluster.KMeans;
import com.mongodb.BasicDBList;
import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import data.CheckinDataset;
import data.WordDataset;
import distribution.Gaussian;
import distribution.Multinomial;
import myutils.ArrayUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Trains a background model that has K states.
 * Each state generate geographical data from Gaussian and keywords from Multinomial.
 * Created by chao on 4/14/15.
 */

public class Background implements Serializable {

    int maxIter;
    int N; // The number of data points.
    int V; // The number of data points.
    int K; // The number of latent states.
    double totalLL;

    double [][] gamma; // This array stores the probability distributions over the latent states for all the data points.
    double [] pi;  // The prior distribution over the latent states.
//    Multinomial[] textModel;
    Gaussian[] geoModel;
//    Gaussian[] temporalModel;
    Multinomial[] itemModel;

    public Background(int maxIter) {
        this.maxIter = maxIter;
    }

//    public Background(DBObject o) {
//        load(o);
//    }

    public void train(CheckinDataset bgd, int K) {
        init(bgd, K);
        double prevLL = totalLL;
        for (int iter = 0; iter < maxIter; iter ++) {
            eStep(bgd);
            mStep(bgd);
            calcTotalLL(bgd);
            System.out.println("Background model finished iteration " + iter + ". Log-likelihood:" + totalLL);
            if(Math.abs(totalLL - prevLL) <= 0.01)
                break;
            prevLL = totalLL;
        }
    }

    /**
     * Step 1: initialize the data, and geo, temporal, and text models.
     */
    private void init(CheckinDataset bgd, int K) {
        this.N = bgd.numPlace();
        this.V = bgd.numWord();
        this.K = K;
        KMeans kMeans = new KMeans(500);
        List<Integer> [] kMeansResults = kMeans.cluster(bgd.getGeoData(), bgd.getTemporalData(), bgd.getWeights(), K);
        initPi(kMeansResults, bgd);
        initGeoModel(kMeansResults, bgd);
//        initTemporalModel(kMeansResults, bgd);
//        initTextModel(kMeansResults, bgd);
        gamma = new double[N][K];
    }

    private void initPi(List<Integer> [] kMeansResults, CheckinDataset bgd) {
        pi = new double [K];
        for(int i=0; i<K; i++) {
            List<Integer> dataIds = kMeansResults[i];
            for(int dataId : dataIds)
              pi[i] += bgd.getWeight(dataId);
        }
        for(int i=0; i<K; i++) {
          pi[i] /= bgd.getWeightedSum();
        }
    }


    private void initGeoModel(List<Integer> [] kMeansResults, CheckinDataset bgd) {
        this.geoModel = new Gaussian[K];
        for(int i=0; i<K; i++) {
            List<Integer> dataIds = kMeansResults[i];
            List<RealVector> clusterData = new ArrayList<RealVector>();
            List<Double> clusterWeights = new ArrayList<Double>();
            for(int dataId : dataIds) {
                clusterData.add(bgd.getGeoDatum(dataId));
                clusterWeights.add(bgd.getWeight(dataId));
            }
            geoModel[i] = new Gaussian();
            geoModel[i].fit(clusterData, clusterWeights);
        }
    }


//    private void initTemporalModel(List<Integer> [] kMeansResults, CheckinDataset bgd) {
//        this.temporalModel = new Gaussian[K];
//        for(int i=0; i<K; i++) {
//            List<Integer> dataIds = kMeansResults[i];
//            List<RealVector> clusterData = new ArrayList<RealVector>();
//            List<Double> clusterWeights = new ArrayList<Double>();
//            for(int dataId : dataIds) {
//                clusterData.add(bgd.getTemporalDatum(dataId));
//                clusterWeights.add(bgd.getWeight(dataId));
//            }
//            temporalModel[i] = new Gaussian();
//            temporalModel[i].fit(clusterData, clusterWeights);
//        }
//    }


//    private void initTextModel(List<Integer> [] kMeansResults, CheckinDataset bgd) {
//        this.textModel = new Multinomial[K];
//        for(int i=0; i<K; i++) {
//            List<Integer> dataIds = kMeansResults[i];
//            List<Map<Integer, Integer>> clusterData = new ArrayList<Map<Integer, Integer>>();
//            List<Double> clusterWeights = new ArrayList<Double>();
//            for(int dataId : dataIds) {
//                clusterData.add(bgd.getTextDatum(dataId));
//                clusterWeights.add(bgd.getWeight(dataId));
//            }
//            textModel[i] = new Multinomial();
////            textModel[i].fit(V, clusterData, clusterWeights);
//        }
//    }

    /**
     * Step 2: learning the parameters using EM: E-Step.
     */
    private void eStep(CheckinDataset bgd) {
        // calc probability in the log domain
        for(int i=0; i<N; i++)
            for (int k = 0; k < K; k++)
                gamma[i][k] = calcLLComponent(bgd.getGeoDatum(i), k);  // p(k, x)
        // normalize
        for(int i=0; i<N; i++)
            ArrayUtils.logNormalize(gamma[i]);
    }



    /**
     * Step 3: learning the parameters using EM: M-Step.
     */
    private void mStep(CheckinDataset bgd) {
        updatePi(bgd);
//        updateTextModel(bgd);
        updateGeoModel(bgd);
//        updateTemporalModel(bgd);
    }

    private void updatePi(CheckinDataset bgd) {
        for(int k=0; k<K; k++) {
            double sum = 0;
            for(int i=0; i<N; i++)
                sum += bgd.getWeight(i) * gamma[i][k];
            pi[k] = sum / bgd.getWeightedSum();
        }
    }

//    private void updateTextModel(CheckinDataset bgd) {
//        for(int k=0; k<K; k++) {
//            List<Double> textWeights = new ArrayList<Double>();
//            for (int i=0; i<N; i++) {
//                textWeights.add( bgd.getWeight(i) * gamma[i][k] );
//            }
//            textModel[k].fit(V, bgd.getTextData(), textWeights);
//        }
//    }

    private void updateGeoModel(CheckinDataset bgd) {
        for(int k=0; k<K; k++) {
            List<Double> geoWeights = new ArrayList<Double>();
            for (int i=0; i<N; i++) {
                geoWeights.add( bgd.getWeight(i) * gamma[i][k] );
            }
            geoModel[k].fit(bgd.getGeoData(), geoWeights);
        }
    }

//    private void updateTemporalModel(CheckinDataset bgd) {
//        for(int k=0; k<K; k++) {
//            List<Double> temporalWeights = new ArrayList<Double>();
//            for (int i=0; i<N; i++) {
//                temporalWeights.add(bgd.getWeight(i) * gamma[i][k]);
//            }
//            temporalModel[k].fit(bgd.getTemporalData(), temporalWeights);
//        }
//    }

    /**
     * Utility functions.
     */
    private void calcTotalLL(CheckinDataset bgd) {
        totalLL = 0;
        for (int i=0; i<N; i++)
            totalLL += calcLL(bgd.getGeoDatum(i), bgd.getTemporalDatum(i));
    }

    public double calcLL(RealVector geoDatum) {
        double [] lnProb = new double [K];
        for (int k=0; k<K; k++)
            lnProb[k] = calcLLComponent(geoDatum, k);
        double maxLnProb = ArrayUtils.max(lnProb);
        for (int k=0; k<K; k++)
            lnProb[k] -= maxLnProb;
        double sum = 0;
        for (int k=0; k<K; k++)
            sum += Math.exp(lnProb[k]);
        if(sum == 0)
            System.out.println("Warning. Sum is 0 when computing log-likelihood for Background.");
        return maxLnProb + Math.log(sum);
    }

    public double calcLL(int itemDatum) {
        double [] lnProb = new double [K];
        for (int k=0; k<K; k++)
            lnProb[k] = calcLLComponent(itemDatum, k);
        double maxLnProb = ArrayUtils.max(lnProb);
        for (int k=0; k<K; k++)
            lnProb[k] -= maxLnProb;
        double sum = 0;
        for (int k=0; k<K; k++)
            sum += Math.exp(lnProb[k]);
        if(sum == 0)
            System.out.println("Warning. Sum is 0 when computing log-likelihood for Background.");
        return maxLnProb + Math.log(sum);
    }

    // Calc ln p(x, k)
    private double calcLLComponent(RealVector geoDatum, int k) {
        double priorProb = Math.log(pi[k]);
        double geoProb = geoModel[k].calcLL(geoDatum);
        return priorProb + geoProb;
    }
    
    private double calcLLComponent(int itemDatum, int k) {
        double priorProb = Math.log(pi[k]);
        double itemProb = itemModel[k].calcLL(itemDatum);
        return priorProb + itemProb;
    }


    // calc ll for a length-2 sequence
    public double calcLL(RealVector geoDatumA, RealVector geoDatumB) {
        double [] lnProb = new double [K];
        for (int k=0; k<K; k++)
            lnProb[k] = calcLLComponentForSeqs(geoDatumA, geoDatumB, k);
        double maxLnProb = ArrayUtils.max(lnProb);
        for (int k=0; k<K; k++)
            lnProb[k] -= maxLnProb;
        double sum = 0;
        for (int k=0; k<K; k++)
            sum += Math.exp(lnProb[k]);
        if(sum == 0)
            System.out.println("Warning. Sum is 0 when computing log-likelihood for Background.");
        return maxLnProb + Math.log(sum);
    }

    public double calcLL(int itemDatumA, int itemDatumB) {
        double [] lnProb = new double [K];
        for (int k=0; k<K; k++)
            lnProb[k] = calcLLComponentForSeqs(itemDatumA, itemDatumB, k);
        double maxLnProb = ArrayUtils.max(lnProb);
        for (int k=0; k<K; k++)
            lnProb[k] -= maxLnProb;
        double sum = 0;
        for (int k=0; k<K; k++)
            sum += Math.exp(lnProb[k]);
        if(sum == 0)
            System.out.println("Warning. Sum is 0 when computing log-likelihood for Background.");
        return maxLnProb + Math.log(sum);
    }

    // Calc ln p(x, k)
    private double calcLLComponentForSeqs(RealVector geoDatumA, RealVector geoDatumB, int k) {
        double priorProb = Math.log(pi[k]);
        double geoProb = geoModel[k].calcLL(geoDatumA);
        double geoProbB = geoModel[k].calcLL(geoDatumB);
        return priorProb + geoProb + geoProbB;
    }
    
    private double calcLLComponentForSeqs(int itemDatumA, int itemDatumB, int k) {
        double priorProb = Math.log(pi[k]);
        double itemProb = itemModel[k].calcLL(itemDatumA);
        double itemProbB = itemModel[k].calcLL(itemDatumB);
        return priorProb + itemProb + itemProbB;
    }

//    public String toString(WordDataset wd) {
//        // Write K.
//        String s = "# K\n";
//        s += K + "\n";
//        // Write Pi.
//        s += "# Pi\n";
//        for(int i=0; i<K; i++)
//            s +=  pi[i] + " ";
//        // Write geo model.
//        s += "\n# geo\n";
//        for(int i=0; i<K; i++) {
//            RealVector mean = geoModel[i].getMean();
//            RealMatrix var = geoModel[i].getVar();
//            s += mean.getEntry(0) + " " + mean.getEntry(1) + " ";
//            s += var.getEntry(0,0) + " " + var.getEntry(0,1) + " " +
//                    var.getEntry(1,0) + " " + var.getEntry(1,1) + "\n";
//        }
//        // Write temporal model.
//        s += "\n# temporal\n";
//        for(int i=0; i<K; i++) {
//            RealVector mean = temporalModel[i].getMean();
//            RealMatrix var = temporalModel[i].getVar();
//            s += mean.getEntry(0) + " ";
//            s += var.getEntry(0, 0) + "\n";
//        }
//        // Write text model.
//        s += "# text\n";
//        for(int i=0; i<K; i++) {
//            s += "------------------------------ State " + i + "------------------------------\n";
//            s += textModel[i].getWordDistribution(wd, 20) + "\n";  // Output the top 20 words.
//        }
//        return s;
//    }

    // Load from a model file.
    public static Background load(String inputFile) throws Exception {
        ObjectInputStream objectinputstream = new ObjectInputStream(new FileInputStream(inputFile));
        Background b = (Background) objectinputstream.readObject();
        objectinputstream.close();
        return b;
    }

    // Serialize
    public void serialize(String serializeFile) throws Exception {
        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(serializeFile));
        oos.writeObject(this);
        oos.close();
    }

//    public void write(WordDataset wd, String outputFile) throws Exception {
//        BufferedWriter bw = new BufferedWriter(new FileWriter(outputFile, false));
//        bw.append(this.toString(wd));
//        bw.close();
//    }
//
//    public DBObject toBson() {
//        DBObject o =  new BasicDBObject()
//                .append("numState", K)
//                .append("numSeq", N)
//                .append("numWord", V)
//                .append("pi", pi);
//        List<DBObject> text = new ArrayList<DBObject>();
//        for(Multinomial m : textModel)
//            text.add(m.toBSon());
//        o.put("textModel", text);
//        List<DBObject> geo = new ArrayList<DBObject>();
//        for(Gaussian g : geoModel)
//            geo.add(g.toBSon());
//        o.put("geoModel", geo);
//        List<DBObject> temporal = new ArrayList<DBObject>();
//        for(Gaussian t : temporalModel)
//            temporal.add(t.toBSon());
//        o.put("temporalModel", temporal);
//        return o;
//    }

//    private void load(DBObject o) {
//        this.K = (Integer) o.get("numState");
//        this.N = (Integer) o.get("numSeq");
//        this.V = (Integer) o.get("numWord");
//
//        BasicDBList piList = (BasicDBList) o.get("pi");
//        this.pi = new double[piList.size()];
//        for(int i=0; i<piList.size(); i++) {
//            this.pi[i] = (Double) piList.get(i);
//        }
//
//        List<DBObject> text = (List<DBObject>) o.get("textModel");
//        this.textModel = new Multinomial[text.size()];
//        for(int i=0; i<text.size(); i++)
//            this.textModel[i] = new Multinomial(text.get(i));
//
//        List<DBObject> geo = (List<DBObject>) o.get("geoModel");
//        this.geoModel = new Gaussian[text.size()];
//        for(int i=0; i<geo.size(); i++)
//            this.geoModel[i] = new Gaussian(geo.get(i));
//
//        List<DBObject> temporal = (List<DBObject>) o.get("temporalModel");
//        this.temporalModel = new Gaussian[text.size()];
//        for(int i=0; i<temporal.size(); i++)
//            this.temporalModel[i] = new Gaussian(temporal.get(i));
//    }

}
