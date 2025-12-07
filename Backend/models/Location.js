import mongoose from "mongoose";

const LocationSchema = new mongoose.Schema(
  {
    camera_name: {
      type: String,
      required: true,
      trim: true,
    },
    rtsp_link: {
      type: String,
      required: true,
    },
    location: {
      type: String,
      required: true,
      trim: true,
    },
  },
  { timestamps: true }
);

export default mongoose.model("Location", LocationSchema);
