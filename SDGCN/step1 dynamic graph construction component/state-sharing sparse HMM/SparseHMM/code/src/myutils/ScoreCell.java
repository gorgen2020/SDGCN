package myutils;

/**
 * Created by chao on 5/4/15.
 */

public class ScoreCell {
    int id;
    double score;
    public ScoreCell(int id, double score) {
        this.id = id;
        this.score = score;
    }
    public int getId() {
        return id;
    }
    public double getScore() {
        return score;
    }
}
