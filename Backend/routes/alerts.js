// routes/alerts.js
import express from 'express';
import mongoose from 'mongoose';
import MissingReport from '../models/MissingReport.js'; // <-- use your actual model file name

const router = express.Router();

// helper: turn whatever person_id is into a clean string
const toStringId = (id) => {
  if (!id) return null;
  if (typeof id === 'string') return id.trim();
  if (typeof id === 'object' && id.toString) return id.toString();
  return null;
};

router.get('/alerts', async (req, res) => {
  try {
    const alertsCollection = mongoose.connection.collection('alerts');

    // 1) get all alerts
    const rawAlerts = await alertsCollection
      .find({})
      .sort({ timestamp: -1 })
      .toArray();

    // 2) collect ONLY valid 24-char hex person_ids
    const stringPersonIds = rawAlerts
      .map(a => toStringId(a.person_id))
      .filter(id => id && /^[0-9a-fA-F]{24}$/.test(id));  // keep only valid ObjectId strings

    const objectIds = stringPersonIds.map(
      id => new mongoose.Types.ObjectId(id)
    );

    // 3) fetch corresponding reports
    const reports = await MissingReport.find({ _id: { $in: objectIds } }).lean();

    const reportMap = new Map(
      reports.map(r => [r._id.toString(), r])
    );

    // 4) build final alerts
    const enrichedAlerts = rawAlerts.map(a => {
      const key = toStringId(a.person_id);        // string or null
      const report = key ? reportMap.get(key) : null;
      const personName = report?.personName || key || 'Unknown person';

      // format timestamp to IST
      let istTime = 'Unknown time';
      if (a.timestamp) {
        istTime = new Date(a.timestamp).toLocaleString('en-UK', {
          timeZone: 'Europe/London',
          year: 'numeric',
          month: 'short',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: true,
        });
      }

      const location = a.location || 'Unknown location';

      const finalMessage =
        a.message ||
        `Person: ${personName} was found at ${location} at ${istTime}`;

      return {
        id: a._id.toString(),
        personId: key,
        personName,
        location,
        time: istTime,
        confidence: a.confidence,
        image_path: a.image_path,
        status: a.status || 'new',
        // message: `Person: ${personName} was found at ${location} at ${istTime}`,
        message: finalMessage,
        type: 'match',
      };
    });

    res.json(enrichedAlerts);
  } catch (err) {
    console.error('Error fetching alerts:', err);
    res.status(500).json({ error: 'Failed to fetch alerts' });
  }
});

export default router;
