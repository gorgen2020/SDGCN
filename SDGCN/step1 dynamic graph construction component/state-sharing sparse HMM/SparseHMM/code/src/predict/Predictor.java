package predict;

import data.Checkin;
import data.PredictionDataset;
import data.Sequence;
import data.SequenceDataset;
import demo.Demo;
import model.ShareHMM;
import myutils.ScoreCell;
import myutils.TopKSearcher;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import JavaExtractor.utils;


public abstract class Predictor {
    List<Double> accuracyList = new ArrayList<Double>();


    int U;
    int[] userNumPred;
    int[][] userNumCorrectK;
    List<Integer> KList;
    
    // Input: a database of length-2 movements;
    public void predict(PredictionDataset mdb, List<Integer> KList, int counter) throws IOException {
    	U = SequenceDataset.getNumUsers();
    	//generate whole array
    	userNumPred = new int[U];
    	
    	//Generate a two-dimensional array of 5 sequences of each element in the total number
    	userNumCorrectK = new int[U][KList.size()];
    	this.KList = KList;
    	

    	
    	if(Demo.underlyingDistribution.equals("2dGaussian")){
    		predictGeo(mdb, KList);	
//    		predictItem(mdb, KList, counter);
		}
		else if(Demo.underlyingDistribution.equals("multinomial")){
			predictItem(mdb, KList, counter);
		}
    	if(Demo.saveUserAcc)
    		writeUserAcc(counter);    	
    }
    
    public void predictGeo(PredictionDataset mdb, List<Integer> KList) {
    	
        int[] numCorrectK = new int[KList.size()];
        int Kmax = KList.get(KList.size()-1);
        
        
      //Defines an array of adjacency matrices
        int[][] adj = new int[SequenceDataset.userIdList.size()][KList.size()];        
        System.out.println(SequenceDataset.userIdList.size());
        
        
        for (int i=0; i<mdb.size(); i++) {
//        	int step = mdb.size()/100;
//        	if(i%step == 0)
//        		System.out.println("prediction ..." + i/step + "%");
            Sequence m = mdb.getSeq(i);
             
            
            int u = SequenceDataset.userIdList.indexOf(m.getUserId());
//            System.out.println(m.getUserId());
            userNumPred[u]++;
            
            Set<Checkin> candidate = mdb.getCands(i);
            TopKSearcher tks = new TopKSearcher();
            tks.init(Kmax);
            for (Checkin p : candidate) {
                ScoreCell sc = calcScore(m, p);
                tks.add(sc);
            }
            ScoreCell [] topKResults = new ScoreCell[Kmax];
            topKResults = tks.getTopKList(topKResults);
            int queueLength = Kmax;
        	//System.out.println("Kmax" + Kmax);
            for (int j = 0; j < Kmax; j++)
            	if(topKResults[j] == null)
            		queueLength--;
        	//System.out.println("queueLength" + queueLength);          
            sort(topKResults, queueLength);           		
            for (int j = 0; j < KList.size(); j++)
            {
          	//System.out.println("topKResults" + topKResults[j].getScore());         	            	
            	int b = KList.get(j);
            	for (int k = 0; k < Math.min(queueLength, KList.get(j)); k++){
            		//System.out.print(topKResults[k].getId()+"");
            		adj[u][j]=topKResults[k].getId();           	            				
               //     System.out.println("a   " + m.getCheckin(1).getId() + "bb" + topKResults[k].getId());        		
	            
            		if (m.getCheckin(1).getId() == topKResults[k].getId()) {            		 		
	            		numCorrectK[j]++;	            		
	            		userNumCorrectK[u][j]++;       		           		
	             //       System.out.println(candidate.size() + " +");
	                } else {
	              //     System.out.println(candidate.size() + " -");
	                }
            	}
            }
        }
        for (int j = 0; j < KList.size(); j++){
        	accuracyList.add((double) numCorrectK[j] / (double) mdb.size());
        }        	
    	
    }

    public void predictItem(PredictionDataset mdb, List<Integer> KList, int counter) {
        int[] numCorrectK = new int[KList.size()];
        int Kmax = KList.get(KList.size()-1);
        
        int[][] adj = new int[SequenceDataset.userIdList.size()][KList.size()]; 
        double[] [] mark = new double[SequenceDataset.userIdList.size()][KList.size()];
        double predict_start=0,predict_stop=0,t=0;
        double sum_time = 0;
                
        predict_start = System.currentTimeMillis();
        
        for (int i=0; i<mdb.size(); i++) {	
        	int step = mdb.size()/100;
        	if(i%step == 0)
        	{
//        		System.out.println("prediction ..." + i/step + "%");
        	}
            Sequence m = mdb.getSeq(i);
            int u = SequenceDataset.userIdList.indexOf(m.getUserId());
            userNumPred[u]++;
            Set<Checkin> candidate = mdb.getCands(i);
//            // use all items for candidates
//            Set<Checkin> candidate = new HashSet<Checkin>();
//            for (int j=0; j<SequenceDataset.getD(); j++) {
//            	candidate.add(new Checkin(j, 0, m.getCheckin(0).getUserId(), j));
//            }
            TopKSearcher tks = new TopKSearcher();
            tks.init(Kmax);
            
            
            predict_start = System.currentTimeMillis();
            for (Checkin p : candidate) {
                ScoreCell sc = calcScore(m, p);
                tks.add(sc);
            }
            predict_stop = System.currentTimeMillis();
            t=predict_stop - predict_start;
            sum_time=sum_time + t;
                          
            ScoreCell [] topKResults = new ScoreCell[Kmax];
            topKResults = tks.getTopKList(topKResults);
            int queueLength = Kmax;
            for (int j = 0; j < Kmax; j++)
            	if(topKResults[j] == null)
            		queueLength--;
            sort(topKResults, queueLength);
            for (int j = 0; j < KList.size(); j++)
            {

           		
            	for (int k = 0; k < Math.min(queueLength, KList.get(j)); k++){
            		
            		adj[u][j]=topKResults[k].getId();
            		mark[u][j]=topKResults[k].getScore();
            		// result get Id is equal to checkin item Id
	            	if (m.getCheckin(1).getItemId() == topKResults[k].getId()) {
	            		numCorrectK[j]++;
	            		userNumCorrectK[u][j]++;
	//                    System.out.println(candidate.size() + " +");
	                } else {
	//                    System.out.println(candidate.size() + " -");
	                }
            	}
            }      
            
        }
        

    
        
        for (int j = 0; j < KList.size(); j++){
        	accuracyList.add((double) numCorrectK[j] / (double) mdb.size());
        }
        
        //save the PDF output
        
        utils record = new utils();
    	for(int i=0; i<SequenceDataset.userIdList.size();i++)
    	{           
//    		System.out.println("hello" + SequenceDataset.userIdList.size());
          		
      		utils.genreateOutputFiles("head" + SequenceDataset.userIdList.get(i),  "C:\\Users\\gorgen\\Desktop\\S3HMM\\result.txt");
      		
      		utils.genreateOutputFiles(SequenceDataset.userIdList.get(i) + ",",  "C:\\Users\\gorgen\\Desktop\\S3HMM\\predict\\result "+ counter + ".txt");   
      		
      		utils.genreateOutputFiles(SequenceDataset.userIdList.get(i) + ",",  "C:\\Users\\gorgen\\Desktop\\S3HMM\\PDF\\result"+ counter + ".txt");      		
    		//System.out.print( "newhead" + SequenceDataset.userIdList.get(i));
    		for(int j=0;j<KList.size();j++)
    		{
    			if(j != (KList.size()-1) )
    			{
	      			utils.genreateOutputFiles("LJB" + adj[i][j] + "PDF" + mark[i][j], "C:\\Users\\gorgen\\Desktop\\S3HMM\\result.txt");
	      			
	      			utils.genreateOutputFiles(adj[i][j] + "," , "C:\\Users\\gorgen\\Desktop\\S3HMM\\predict\\result "+ counter + ".txt");  
	
	      			utils.genreateOutputFiles( mark[i][j] + "," , "C:\\Users\\gorgen\\Desktop\\S3HMM\\PDF\\result"+ counter + ".txt");  
	    			//System.out.print("LJB" + adj[i][j]);
	    			//System.out.print("PDF" + mark[i][j]);    				
    			}
    			else
    			{
	      			utils.genreateOutputFiles("LJB" + adj[i][j] + "PDF" + mark[i][j], "C:\\Users\\gorgen\\Desktop\\S3HMM\\result.txt");
	      			
	      			utils.genreateOutputFiles(adj[i][j] + "" , "C:\\Users\\gorgen\\Desktop\\S3HMM\\predict\\result "+ counter + ".txt");  
	
	      			utils.genreateOutputFiles(mark[i][j] + "" , "C:\\Users\\gorgen\\Desktop\\S3HMM\\PDF\\result"+ counter + ".txt");  

    			}
    		}
    		//System.out.println(" ");
  			utils.genreateOutputFiles("\n",  "C:\\Users\\gorgen\\Desktop\\S3HMM\\result.txt");
  			
  			utils.genreateOutputFiles("\n",  "C:\\Users\\gorgen\\Desktop\\S3HMM\\predict\\result "+ counter + ".txt");
 
  			utils.genreateOutputFiles("\n",  "C:\\Users\\gorgen\\Desktop\\S3HMM\\PDF\\result"+ counter + ".txt");
    	}
    	        	    	
    }

    public abstract ScoreCell calcScore(Sequence m, Checkin p);

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
    
    public boolean isCorrect(Sequence m, ScoreCell [] topKResult) {
        int groundTruth = m.getCheckin(1).getId();
        for (int i = 0; i < topKResult.length; i++)
            if (groundTruth == topKResult[i].getId())
                return  true;
        return false;
    }

    public List<Double> getAccuracy() {
        return accuracyList;
    }
    
    void writeUserAcc(int counter) throws IOException {
    	FileWriter fw;
    	String dir = "../result/txt/userPred/";
    	File fileDir = new File(dir);
		fileDir.mkdirs();
    	fw = new FileWriter(dir + Demo.dataset + Demo.method + counter + ".txt"); 
    	// first line: title
		String title = "userNumPred ";
		for (int j = 0; j < userNumCorrectK[0].length; j++) {
			title += "top" + KList.get(j) + " ";
		}
		fw.write(title + "\n");
		// then one user per line
    	for (int u = 0; u < U; u++) {
    		fw.write(String.valueOf(userNumPred[u]));fw.write(" ");
    		for (int j = 0; j < userNumCorrectK[0].length; j++) {
    			fw.write(String.valueOf(userNumCorrectK[u][j]));fw.write(" ");
    		}
			fw.write("\n");
    	}
    	fw.close();
    }
}
