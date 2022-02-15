package data;

/* *
 * This class represents a checkin sequence of a user.
 */

import java.io.Serializable;
import java.util.*;

public class Sequence implements Serializable {

	Long userId;

	// a list of checkins for the user
	List<Checkin> checkins = null;

	public Sequence() {
	}

	public Sequence(Long userId) {
		this.userId = userId;
		checkins = new ArrayList<Checkin>();
	}

	public Sequence(Long userId, List<Checkin> checkins) {
		this.userId = userId;
		this.checkins = checkins;
	}

	public List<Checkin> getCheckins() {
		return checkins;
	}

	public Checkin getCheckin(int index) {
		return checkins.get(index);
	}

	public int size() {
		return checkins.size();
	}

	public void addCheckin(Checkin c) {
		checkins.add(c);
	}

	public void sortCheckins() {
		Collections.sort(checkins, new Comparator<Checkin>() {
			public int compare(Checkin c1, Checkin c2) {
				if (c1.getTimestamp() - c2.getTimestamp() > 0)
					return 1;
				else if (c1.getTimestamp() - c2.getTimestamp() == 0)
					return 0;
				else
					return -1;
			}
		});
	}

	public long getUserId() {
		return userId;
	}

	public Sequence copy() {
		Sequence res = new Sequence(this.userId);
		for (Checkin c : checkins) {
			res.addCheckin(c.copy());
		}
		return res;
	}
}
