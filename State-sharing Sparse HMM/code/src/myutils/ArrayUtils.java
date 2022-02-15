package myutils;

import java.util.List;
import java.util.Random;

/**
 * Created by chao on 4/20/15.
 */
public class ArrayUtils {

    static Random r = new Random(100);

    // Find the max value of an array.
    public static double max(double [] data) {
        if (data.length == 0) {
            System.out.println("Error when finding the max value. Array length is 0!");
            System.exit(1);
        }
        double maxValue = data[0];
        for (int i=0; i<data.length; i++) {
            if (data[i] > maxValue)
                maxValue = data[i];
        }
        return maxValue;
    }
    
    // Find the idx of max value of an array.
    public static int maxIdx(double [] data) {
        if (data.length == 0) {
            System.out.println("Error when finding the max value. Array length is 0!");
            System.exit(1);
        }
        double maxValue = data[0];
        int maxIdx = 0;
        for (int i=0; i<data.length; i++) {
            if (data[i] > maxValue){
                maxValue = data[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // Find the min value of an array.
    public static double min(double [] data) {
        if (data.length == 0) {
            System.out.println("Error when finding the min value. Array length is 0!");
            System.exit(1);
        }
        double minValue = data[0];
        for (int i=0; i<data.length; i++) {
            if (data[i] < minValue)
                minValue = data[i];
        }
        return minValue;
    }
    
 // Find the idx of min value of an array.
    public static int minIdx(double [] data) {
        if (data.length == 0) {
            System.out.println("Error when finding the min value. Array length is 0!");
            System.exit(1);
        }
        double minValue = data[0];
        int minIdx = 0;
        for (int i=0; i<data.length; i++) {
            if (data[i] < minValue){
                minValue = data[i];
                minIdx = i;
            }
        }
        return minIdx;
    }

    // Find the sum of the array
    public static double sum(double [] data) {
        if (data.length == 0) {
            System.out.println("Error when finding the sum. Array length is 0!");
            System.exit(1);
        }
        double sumValue = 0;
        for (int i=0; i<data.length; i++) {
            sumValue += data[i];
        }
        return sumValue;
    }

    // Find the sum of exp of the array
    public static double expSum(double [] data) {
        double sumValue = 0;
        for (int i=0; i<data.length; i++) {
            sumValue += Math.exp(data[i]);
        }
        return sumValue;
    }

    // Find the sum of exp of the array, and then take the log
    public static double sumExpLog(double [] data) {
        double maxValue = max(data);
        double sumValue = 0;
        for (int i=0; i<data.length; i++)
            sumValue += Math.exp(data[i] - maxValue);
        
//        if(Double.isNaN(Math.log(sumValue) + maxValue))
//        	System.out.println("sumexplog result is nan!");
        	
        return Math.log(sumValue) + maxValue;
    }

    // Normalize the array by sum
    public static void normalize(double [] data) {
        if (data.length == 0) {
            System.out.println("Error when normalizing. Array length is 0!");
            System.exit(1);
        }
        double sumValue = sum(data);
        if (sumValue == 0) {
           // System.out.println("Warning: sum of the elements is 0 when normalizing!");
            return;
        }
        for (int i=0; i<data.length; i++)
            data[i] /= sumValue;
    }


    // Normalize the array to the range [0, 1]
    public static void normalizeZeroOne(double [] data) {
        if (data.length == 0) {
            System.out.println("Error when zero-one normalizing. Array length is 0!");
            System.exit(1);
        }
        double maxValue = max(data);
        double minValue = min(data);
        if(minValue == maxValue)    return;
        for (int i=0; i<data.length; i++)
            data[i] = (data[i] - minValue) / (maxValue - minValue);
    }

    // Input: an array in the log domain; Output: the ratio in the exp domain
    public static void logNormalize(double[] data) {
        if (data.length == 0) {
            System.out.println("Error when doing log-sum-exp. Array length is 0!");
            System.exit(1);
        }
        double maxValue = max(data);
        for (int i=0; i<data.length; i++){
            data[i] = Math.exp(data[i] - maxValue);
//            if(Double.isNaN(data[i]))
//				System.out.println("data[i] is NAN. i = " + i);
        }
        normalize(data);
    }


    // sum of a list
    public static double sum(List<Double> data) {
        double ret = 0;
        for (Double d : data)
            ret += d;
        return ret;
    }

    // calc accuracy
    public static double calcAccuracy(List<Integer> groundTruth, List<Integer> predicted) {
        if (groundTruth.size() != predicted.size()) {
            System.out.println("Error, the ground truth and predicted data do not have equal length!");
            System.exit(1);
        }
        System.out.println(groundTruth);
        System.out.println(predicted);
        int denominator = groundTruth.size();
        int numerator = 0;
        for (int i=0; i<groundTruth.size(); i++) {
            if (groundTruth.get(i).intValue() == predicted.get(i).intValue())
                numerator += 1;
        }
        return (double) numerator / (double) denominator;
    }


    // Generate K distinct random numbers in [0,n-1]
    public static int[] genKRandomNumbers(int n, int k) {
        int[] completeArray = new int[n];
        for(int i=0; i<n; i++) {
            completeArray[i] = i;
        }
        int[] result = new int[k];
        int bound = n;
        for(int i=0; i<k; i++) {
            int randNum = r.nextInt( bound ); //generate a random integer between 0~bound-1
            result[i] = completeArray[ randNum ];
            completeArray[randNum] = completeArray[ bound-1 ];
            completeArray[bound-1] = result[i];
            bound --;
        }
        return result;
    }
}
