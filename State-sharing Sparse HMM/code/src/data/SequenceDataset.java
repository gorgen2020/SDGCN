package data;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.io.*;
import java.util.*;



public class SequenceDataset {
	int checkinId = 0;
	String dataset;
	String underlyingDistribution;
//	List<String> users;
	// training data uniform data structure
	List<Sequence> trainseqs = new ArrayList<Sequence>();
	
	// for original data structure
	List<RealVector> geoData = new ArrayList<RealVector>(); // The geographical data for the R seqeunces, length 2R
	List<Integer> itemData = new ArrayList<Integer>(); // The item data for the R seqeunces, length 2R
	List<RealVector> temporalData = new ArrayList<RealVector>(); // The temporal data for the R seqeunces, length 2R
	static int D = 0; // the number of items
	
	// User-Sequence
	public static HashMap<Long, HashSet<Integer>> user2seqs = new HashMap<Long, HashSet<Integer>>();
	public static HashMap<Long, HashSet<Integer>> user2pnts = new HashMap<Long, HashSet<Integer>>();
	public static List<Object> userIdList;
	public static Long[] pnts2user;
	public static int[] pnts2useridx;
	
	public static int f=0;
	// test data
	double testRatio;
	List<Sequence> testSeqs = new ArrayList<Sequence>();
	
	public void load(String sequenceFile, double testRatio, String dataset, String underlyingDistribution, boolean filterTest) throws IOException {
		this.dataset = dataset;
		this.underlyingDistribution = underlyingDistribution;
		this.testRatio = testRatio;
		List<Sequence> allSeqs = new ArrayList<Sequence>();
		allSeqs.clear();
		System.out.println(sequenceFile);
		BufferedReader br = new BufferedReader(new FileReader(sequenceFile));
		while (true) {
			f++;
//			System.out.println(f);
			String line = br.readLine();
			if (line == null)
				break;
			Sequence seq = parseSequence(line, dataset);				
			allSeqs.add(seq);
		}
		br.close();

		trainseqs.clear();
		testSeqs.clear();
		
		
		Collections.shuffle(allSeqs, new Random(1));
		trainseqs = allSeqs.subList(0, (int) (allSeqs.size() * (1 - testRatio)));
		testSeqs = allSeqs.subList((int) (allSeqs.size() * (1 - testRatio)), allSeqs.size());
		if (filterTest) {
			filterTestSeqs(allSeqs);
		}
		
		if(userIdList == null)
		{}
		else
		{
			user2seqs.clear();
			user2pnts.clear();
		}


		
		calcUser2seqs();
//		System.out.println("userIdList.size(): " + userIdList.size());

		// Geo or item  and temporal data.
		if(underlyingDistribution.equals("2dGaussian")){
			geoData = new ArrayList<RealVector>();
		}
		else if(underlyingDistribution.equals("multinomial")){
			itemData = new ArrayList<Integer>();
		}
		temporalData = new ArrayList<RealVector>();
		for (Sequence sequence : trainseqs) {
			if (sequence.size() != 2) {
				System.out.println("Warning! The sequence's length is not 2.");
			}
			List<Checkin> checkins = sequence.getCheckins();
			for (Checkin c : checkins) {
				if(underlyingDistribution.equals("2dGaussian")){
								      	
					geoData.add(c.getLocation().toRealVector());
				}
				else if(underlyingDistribution.equals("multinomial")){
					itemData.add(c.getItemId());
					// count how many items
					if(D < c.getItemId() + 1)
						D = c.getItemId() + 1;
				}
				temporalData.add(new ArrayRealVector(new double[] { c.getTimestamp() % 1440 })); // get the minutes of the timestamp.
			}
		}
		System.out.println("Loading sequences.txt finished.");
	}
	
	public void calcUser2seqs() {
		int R = trainseqs.size();
		pnts2user = new Long[R * 2];
		pnts2useridx = new int[R * 2];
		for (int i = 0; i < size(); i++) {
			Sequence seq = trainseqs.get(i);
			long user = seq.getUserId();
			if (!user2seqs.containsKey(user)) {
				user2seqs.put(user, new HashSet<Integer>());
				user2pnts.put(user, new HashSet<Integer>());
			}
			user2seqs.get(user).add(i);
			user2pnts.get(user).add(i * 2);
			user2pnts.get(user).add(i * 2 + 1);
			pnts2user[i * 2] = user;
			pnts2user[i * 2 + 1] = user;
		}
		userIdList = Arrays.asList(user2seqs.keySet().toArray());
		for (int i = 0; i < trainseqs.size(); i++) {
			pnts2useridx[i * 2] = userIdList.indexOf(pnts2user[i * 2]);
			pnts2useridx[i * 2 + 1] = pnts2useridx[i * 2];
		}
//		this.U = user2seqs.size();
//		setWeight(weight);
	}
	
	private void filterTestSeqs(List<Sequence> allSeqs) {
		HashMap<Long, HashSet<Integer>> user2seqs = new HashMap<Long, HashSet<Integer>>();
		for (int i = 0; i < trainseqs.size(); i++) {
			Sequence seq = trainseqs.get(i);
			long user = seq.getUserId();
			if (!user2seqs.containsKey(user)) {
				user2seqs.put(user, new HashSet<Integer>());
			}
			user2seqs.get(user).add(i);
		}
		testSeqs = new ArrayList<Sequence>();
		for (int i = (int) (allSeqs.size() * (1 - testRatio)); i < allSeqs.size(); ++i) {
			Sequence seq = allSeqs.get(i);
			long user = seq.getUserId();
			if (user2seqs.containsKey(user)) {
				testSeqs.add(seq);
			}
		}
//		System.out.println("filtered testSeqs size: " + testSeqs.size());
	}

	// add training seq
	public void addSequence(Sequence s) {
		this.trainseqs.add(s);
	}

	// add test seq
	public void addTestSequence(Sequence s) {
		this.testSeqs.add(s);
	}

	public void addGeoDatum(RealVector rv) {
		this.geoData.add(rv);
	}
	
	public void addItemDatum(int rv) {
		this.itemData.add(rv);
	}
	
	public void addTemporalDatum(RealVector rv) {
		this.temporalData.add(rv);
	}

	public void setTestRatio(double testRatio) {
		this.testRatio = testRatio;
	}

	// Each line contains: checkin Id, userId, placeid, timestamp, message
	private Sequence parseSequence(String line) {
		String[] items = line.split(",");
		Checkin start = toCheckin(Arrays.copyOfRange(items, 0, items.length / 2));
		Checkin end = toCheckin(Arrays.copyOfRange(items, items.length / 2, items.length));
		Long userId = start.getUserId();
		Sequence seq = new Sequence(userId);
		seq.addCheckin(start);
		seq.addCheckin(end);
		return seq;
	}
	
	private Sequence parseSequence(String line, String dataset) {
		String[] items = line.split(",");
		Checkin start = toCheckin(Arrays.copyOfRange(items, 0, items.length / 2));
		Checkin end = toCheckin(Arrays.copyOfRange(items, items.length / 2, items.length));
		Long userId = start.getUserId();
		Sequence seq = new Sequence(userId);
		seq.addCheckin(start);
		seq.addCheckin(end);
		/*
		System.out.println("line" + line);		
		System.out.println("items" + items);		
		System.out.println("尝试打印一条start信息表，userId" + userId + "type" +start.type + "checkinId" + start.checkinId + "timestamp" +start.timestamp 
				+ "itemId" +start.itemId + "location-x" +start.location.lat	+ "location-y" +start.location.lng);
		System.out.println("尝试打印一条end信息表，userId" + userId + "type" +end.type + "checkinId" + end.checkinId + "timestamp" +end.timestamp 
				+ "itemId" +end.itemId + "location-x" +end.location.lat	+ "location-y" +end.location.lng);
*/
		
		return seq;
	}

	private Checkin toCheckin(String[] items) {
		if (items.length < 6) {
			System.out.println("Error when parsing checkins.");
			return null;
		}

		
		int checkinId = Integer.parseInt(items[0]);
		int timestamp = Integer.parseInt(items[1]);
		Long userId = Long.parseLong(items[2]);
		Checkin checkin = null;
		if(underlyingDistribution.equals("2dGaussian")){
			double lat = Double.parseDouble(items[3]);
			double lng = Double.parseDouble(items[4]);
			checkin = new Checkin(checkinId, timestamp, userId, lat, lng);
		}
		else if(underlyingDistribution.equals("multinomial")){
			int itemId = Integer.parseInt(items[5]);
			checkin = new Checkin(checkinId, timestamp, userId, itemId);
		}
		return checkin;
	}

	private Map<Integer, Integer> parseMessage(String s) {
		Map<Integer, Integer> message = new HashMap<Integer, Integer>();
		String[] items = s.split("\\s");
		if (items.length == 0) {
			System.out.println("Warning! Checkin has no message.");
		}
		for (int i = 0; i < items.length; i++) {
			int wordId = Integer.parseInt(items[i]);
			int oldCnt = message.containsKey(wordId) ? message.get(wordId) : 0;
			message.put(wordId, oldCnt + 1);
		}
		return message;
	}

	public List<Sequence> getSequences() {
		return trainseqs;
	}

	public List<RealVector> getGeoData() {
		return geoData;
	}
	
	public List<Integer> getItemData() {
		return itemData;
	}

	public List<RealVector> getTemporalData() {
		return temporalData;
	}

	public RealVector getGeoDatum(int index) {
		return geoData.get(index);
	}
	
	public int getItemDatum(int index) {
		return itemData.get(index);
	}

	public RealVector getTemporalDatum(int index) {
		return temporalData.get(index);
	}

	public Sequence getSequence(int i) {
		return trainseqs.get(i);
	}
	
	public static int getD() {
		return D;
	}
	
	public static int getNumUsers() {
		return userIdList.size();
	}
	
	public int size() {
		return trainseqs.size();
	}

	public PredictionDataset extractTestData() throws Exception {
		return new PredictionDataset(testSeqs);
	}

	public SequenceDataset getCopy() {
		SequenceDataset copiedDataSet = new SequenceDataset();
		for (Sequence s : trainseqs) {
			copiedDataSet.addSequence(s.copy());
		}
		for (Sequence s : testSeqs) {
			copiedDataSet.addTestSequence(s.copy());
		}
		for (RealVector rv : geoData) {
			copiedDataSet.addGeoDatum(rv);
		}
		for (int rv : itemData) {
			copiedDataSet.addItemDatum(rv);
		}
		for (RealVector rv : temporalData) {
			copiedDataSet.addTemporalDatum(rv);
		}
		copiedDataSet.setTestRatio(this.testRatio);
		return copiedDataSet;
	}
}
