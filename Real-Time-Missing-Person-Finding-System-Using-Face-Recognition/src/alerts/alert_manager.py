
import os
import time
import logging
from datetime import datetime
try:
    from pymongo import MongoClient
    from bson.objectid import ObjectId
    MONGO_AVAILABLE = True
except ImportError:
    MONGO_AVAILABLE = False

class AlertManager:
    def __init__(self, mongo_uri, db_name="test", collection_name="alerts", cooldown_seconds=300):
        self.logger = logging.getLogger("AlertManager")
        self.cooldown_seconds = cooldown_seconds
        self.last_alert_time = {}  # {person_id: timestamp}
        self.collection = None
        self.reports_collection = None
        
        if MONGO_AVAILABLE:
            try:
                self.client = MongoClient(mongo_uri)
                self.db = self.client[db_name]
                self.collection = self.db[collection_name]
                self.reports_collection = self.db["missingreports"]
                self.logger.info(f"Connected to MongoDB: {db_name}.{collection_name}")
            except Exception as e:
                self.logger.error(f"Failed to connect to MongoDB: {e}")
        else:
            self.logger.warning("pymongo not installed. Alerts will not be sent.")

    def check_cooldown(self, person_id):
        """Returns True if alert can be sent (cooldown passed or new person)."""
        now = time.time()
        last_time = self.last_alert_time.get(person_id, 0)
        
        if now - last_time > self.cooldown_seconds:
            return True
        return False
        
    def get_person_name(self, person_id):
        """Fetches name from missingreports using _id."""
        if not self.reports_collection or not person_id:
            return None
        
        try:
            # Check if person_id is a valid ObjectId string
            if ObjectId.is_valid(person_id):
                oid = ObjectId(person_id)
                self.logger.debug(f"Querying missingreports for _id: {oid}")
                report = self.reports_collection.find_one({"_id": oid})
                
                if report:
                    self.logger.debug(f"Found report: {report}")
                    if "personName" in report:
                        return report["personName"]
                    else:
                        self.logger.warning(f"Report found for {person_id} but field 'personName' is missing.")
                else:
                    self.logger.warning(f"No report found in missingreports for _id: {oid}")
            else:
                self.logger.warning(f"Invalid ObjectId format: {person_id}")
        except Exception as e:
            self.logger.warning(f"Failed to fetch name for {person_id}: {e}")
        return None

    def send_alert(self, person_id, location="Unknown", confidence=0.0, image_path="", message=None):
        """Sends an alert to MongoDB."""
        if not self.collection:
            return

        # Cooldown check only for person alerts, maybe skip for CROWD_ALERT or handle separately?
        if not self.check_cooldown(person_id):
            self.logger.debug(f"Alert cooldown active for {person_id}. Skipping.")
            return

        # Fetch Name if not CROWD_ALERT
        person_name = person_id
        if person_id != "CROWD_ALERT":
            fetched_name = self.get_person_name(person_id)
            if fetched_name:
                person_name = fetched_name
        
        # Prepare Message
        timestamp = datetime.now()
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        final_message = message
        if not final_message:
            if person_id == "CROWD_ALERT":
                 final_message = f"High crowd density detected at {location} on {formatted_time}"
            else:
                 final_message = f"Person: {person_name} was detected at {location} on {formatted_time}"

        alert_doc = {
            "person_id": person_id,
            "person_name": person_name, # Added field for easy access
            "location": location,
            "timestamp": timestamp,
            "confidence": float(confidence),
            "image_path": image_path,
            "status": "new",
            "message": final_message
        }

        try:
            self.collection.insert_one(alert_doc)
            self.logger.info(f"Alert sent: {final_message}")
            self.last_alert_time[person_id] = time.time()
        except Exception as e:
            self.logger.error(f"Failed to insert alert: {e}")
