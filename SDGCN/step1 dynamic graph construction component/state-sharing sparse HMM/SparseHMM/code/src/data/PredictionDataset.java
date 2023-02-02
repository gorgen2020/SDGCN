package data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import demo.Demo;

/**
 * Created by chao on 12/14/15.
 */
public class PredictionDataset {
    List<Sequence> sequences;
    List<Set<Checkin>> candidates;

    public PredictionDataset(List<Sequence> sequences) {
        this.sequences = sequences;
    }

    public void genCandidates(double distThre, double timeThre) {
        candidates = new ArrayList<Set<Checkin>>();
        if(Demo.underlyingDistribution.equals("2dGaussian"))   {      	
	        for (Sequence seq : sequences) {
	        	Checkin a = seq.getCheckin(1);
	            candidates.add(getNeighbors(seq.getCheckin(1), distThre, timeThre));
	        }
        }
        else if(Demo.underlyingDistribution.equals("multinomial")){
        	for (Sequence seq : sequences) {
	            candidates.add(getNeighbors(seq.getCheckin(1), distThre, timeThre));
	        }
        }
    }

    private Set<Checkin> getNeighbors(Checkin c, double distThre, double timeThre) {
        Set<Checkin> ret = new HashSet<Checkin>();
        // avoid repeated checkins with the same item id
        List<Integer> itemIds = new ArrayList<Integer>();
        for (Sequence seq : sequences) {
        	// select the candidates based on spatial and temporal condition
            for (Checkin cand : seq.getCheckins()) {
                if(Demo.underlyingDistribution.equals("2dGaussian"))   {      	
                	if (c.getLocation().calcGeographicDist(cand.getLocation()) <= distThre &&
                            Math.abs(cand.getTimestamp() - c.getTimestamp()) <= timeThre)
                        ret.add(cand);
                }
                else if(Demo.underlyingDistribution.equals("multinomial")){
                	if (Math.abs(cand.getTimestamp() - c.getTimestamp()) <= timeThre)
                		if(!itemIds.contains(cand.itemId)){
	                		itemIds.add(cand.itemId);
	                        ret.add(cand);
                		}
                }
            }
        }
      //  System.out.print('a');
        return ret;
    }

    public int size() {
        return sequences.size();
    }

    public Sequence getSeq(int i) {
        return sequences.get(i);
    }

    public Set<Checkin> getCands(int i) {
        return candidates.get(i);
    }

}
