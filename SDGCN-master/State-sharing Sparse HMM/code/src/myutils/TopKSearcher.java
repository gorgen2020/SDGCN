package myutils;

import java.util.Comparator;
import java.util.PriorityQueue;

/**
 * Created by chao on 5/3/15.
 */
public class TopKSearcher {

    int K;
    PriorityQueue <ScoreCell> minHeap;

    public void init(int K) {
        this.K = K;
        minHeap = new PriorityQueue(K, //set the initial size to K
                new Comparator<ScoreCell>() {
                    public int compare(ScoreCell c1, ScoreCell c2) {
                        // Ascending order of score
                        if(c1.getScore() - c2.getScore() < 0)
                            return -1;
                        else if(c1.getScore() - c2.getScore() > 0)
                            return 1;
                        else
                            return 0;
                    }
                }
        );
    }

    public void add(ScoreCell sc) {
        if (minHeap.size() < K)
            minHeap.offer(sc);
        else if (minHeap.peek().getScore() < sc.getScore()) {
            minHeap.poll();
            minHeap.offer(sc);
        }
    }
    
    public void sort(ScoreCell[] topKResults, int queueLength){
    	ScoreCell swap;
    	for (int i = 0; i < Math.min(topKResults.length-1, queueLength); i++)
    		for (int j = i+1; j < Math.min(topKResults.length, queueLength); j++)
    			if(topKResults[i].getScore() < topKResults[j].getScore()){
    				swap = topKResults[i];
    				topKResults[i] = topKResults[j];
    				topKResults[j] = swap;
    			}
    }

    public ScoreCell [] getTopKList(ScoreCell [] scArray) {
        return minHeap.toArray(scArray);
    }
}


