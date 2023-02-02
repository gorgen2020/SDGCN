package data;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

public class Checkin implements Serializable {

  String type;
  int checkinId;
  Long userId;
  Location location;
  int timestamp;
  int itemId; // music, goods, or other discrete items
//  Map<Integer, Integer> message;  // Key: word, Value:count

  public Checkin(int checkinId, int timestamp, Long userId, double lat, double lng) {
	this.type = "geo";
	this.checkinId = checkinId;
    this.timestamp = timestamp;
    this.userId = userId;
    this.location = new Location(lng, lat);
    
  }

  public Checkin(int checkinId, int timestamp, Long userId, int itemid) {
	    this.type = "discrete";
	    this.checkinId = checkinId;
	    this.timestamp = timestamp;
	    this.userId = userId;
	    this.itemId = itemid;
	  }
  
  public Checkin(String type, int checkinId, int timestamp, Long userId, double lat, double lng, int item) {
	    this.type = type;
	    this.checkinId = checkinId;
	    this.timestamp = timestamp;
	    this.userId = userId;
	    this.timestamp = timestamp;
	    switch(type){
	    case("geo"): this.location = new Location(lng, lat); break;
	    case("discrete"): this.itemId = item; break;
	    default: System.out.println("Wrong type name for checkin: " + type);
	    }
	  }
  
  public String getType() {
	    return type;
	  }
  
  public int getId() {
    return checkinId;
  }

  public Long getUserId() {
    return userId;
  }

  public Location getLocation() {
    return location;
  }

  public int getTimestamp() {
    return timestamp;
  }

  public int getItemId() {
	    return itemId;
  }

  @Override
  public int hashCode() {
    return new Integer(checkinId).hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof Checkin))
      return false;
    if (obj == this)
      return true;
    return checkinId == ((Checkin) obj).getId();
  }

  public Checkin copy() {
    Checkin res = new Checkin(type, checkinId, timestamp, userId, location.getLat(), location.getLng(), itemId);
    return res;
  }

}
