import mongoose from "mongoose";
import Location from "./models/Location.js";
import dotenv from "dotenv";

dotenv.config(); // loads MONGO_URI from .env

async function seed() {
  try {
    await mongoose.connect(process.env.MONGO_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });

    console.log("Connected to MongoDB");

    // 5 demo data
    const demoData = [
      {
        camera_name: "Camera 1",
        rtsp_link: "rtsp://username:password@192.168.1.3:554/stream1",
        location: "Location 1"
      },
      {
        camera_name: "Camera 2",
        rtsp_link: "rtsp://username:password@192.168.2.3:554/stream2",
        location: "Location 2"
      },
      {
        camera_name: "Camera 3",
        rtsp_link: "rtsp://username:password@192.168.3.3:554/stream3",
        location: "Location 3"
      },
      {
        camera_name: "Camera 4",
        rtsp_link: "rtsp://username:password@192.168.4.3:554/stream4",
        location: "Location 4"
      },
      {
        camera_name: "Camera 5",
        rtsp_link: "rtsp://username:password@192.168.5.3:554/stream5",
        location: "Location 5"
      }
    ];

    // Insert many documents
    const result = await Location.insertMany(demoData);

    console.log("Inserted demo documents:");
    console.log(result);

    await mongoose.disconnect();
    console.log("Disconnected");

  } catch (err) {
    console.error("Error:", err);
  }
}

seed();
