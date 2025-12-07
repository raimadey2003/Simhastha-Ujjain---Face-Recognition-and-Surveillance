import express from 'express';
import MissingReport from '../models/MissingReport.js';
import multer from 'multer';
// import path from 'path';
// import { fileURLToPath } from 'url';

const router = express.Router();
// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// 🔹 Use memory storage so file.buffer is available
const storage = multer.memoryStorage();
const upload = multer({ storage });

// Configure multer for file uploads
// const storage = multer.diskStorage({
//   destination: (req, file, cb) => {
//     cb(null, path.join(__dirname, '../uploads'));
//   },
//   filename: (req, file, cb) => {
//     cb(null, Date.now() + '-' + file.originalname);
//   }
// });
// const upload = multer({ storage });



// Save files in the 'uploads' folder
// const storage = multer.diskStorage({
//   destination: (req, file, cb) => {
//     cb(null, 'uploads/');
//   },
//   filename: (req, file, cb) => {
//     const uniqueSuffix = Date.now() + '-' + file.originalname;
//     cb(null, uniqueSuffix); // store only filename, not full path
//   }
// });

// const upload = multer({ storage });


// Create a new missing report
// router.post('/', upload.array('photos', 5), async (req, res) => {
//   try {
//     const { reporterName, reporterPhone, reporterRelation, personName, personAge, personGender, personHeight, personClothing, lastSeenLocation, lastSeenTime, description } = req.body;
    
//     const photos = req.files ? req.files.map(file => file.filename) : [];

//     // assume req.userId comes from authentication middleware
//     const newReport = new MissingReport({
//       reporterName,
//       reporterPhone,
//       reporterRelation,
//       personName,
//       personAge,
//       personGender,
//       personHeight,
//       personClothing,
//       lastSeenLocation,
//       lastSeenTime,
//       description,
//       photos,
//       user: req.userId // 🔹 link to logged-in user
//     });

//     await newReport.save();
//     res.status(201).json({ message: 'Missing report submitted successfully!', report: newReport });
//   } catch (err) {
//     console.error(err);
//     res.status(500).json({ message: 'Server error' });
//   }
// });


import authMiddleware from '../middleware/auth.js';  // <-- protect route

// Submit Missing Report
// router.post('/', authMiddleware, upload.array('photos', 5), async (req, res) => {
//   try {
//     const reportData = req.body;

//     if (req.files) {
//       reportData.photos = req.files.map(file => file.filename);
//     }

//     // Attach logged-in user (from auth middleware)
//     reportData.user = req.user.id;

//     const report = new MissingReport(reportData);
//     await report.save();

//     res.json({ message: 'Report submitted', report });
//   } catch (err) {
//     console.error("Error submitting report:", err);
//     res.status(500).json({ error: "Failed to submit report" });
//   }
// });

router.post('/', authMiddleware, upload.array('photos', 5), async (req, res) => {
  try {
    const reportData = req.body;

    // 🔹 Convert uploaded files into Blob objects for MongoDB
    if (req.files && req.files.length > 0) {
      reportData.photos = req.files.map(file => ({
        data: file.buffer,         // binary data
        contentType: file.mimetype // e.g. 'image/jpeg'
      }));
    } else {
      reportData.photos = [];
    }

    // Attach logged-in user (from auth middleware)
    reportData.user = req.user.id;

    const report = new MissingReport(reportData);
    await report.save();

    res.json({ message: 'Report submitted', report });
  } catch (err) {
    console.error("Error submitting report:", err);
    res.status(500).json({ error: "Failed to submit report" });
  }
});



// Get all missing reports
router.get('/', async (req, res) => {
  try {
    const reports = await MissingReport.find().populate('user', 'fullName email');
    res.json(reports);
  } catch (err) {
    res.status(500).json({ message: 'Server error' });
  }
});

// Get reports for a specific user (protected route)
router.get('/my-reports', authMiddleware, async (req, res) => {
  try {
    const reports = await MissingReport.find({ user: req.user.id })
      .sort({ createdAt: -1 }); // Most recent first
    res.json(reports);
  } catch (err) {
    console.error("Error fetching user reports:", err);
    res.status(500).json({ error: "Failed to fetch reports" });
  }
});

// Get a specific photo of a report by index
router.get('/:reportId/photos/:index', async (req, res) => {
  try {
    const { reportId, index } = req.params;

    const report = await MissingReport.findById(reportId);
    if (!report) {
      return res.status(404).json({ error: 'Report not found' });
    }

    const photo = report.photos[index];
    if (!photo) {
      return res.status(404).json({ error: 'Photo not found' });
    }

    // Tell browser what type of file this is (jpeg/png/etc)
    res.contentType(photo.contentType);
    // Send the raw bytes
    res.send(photo.data);
  } catch (err) {
    console.error('Error fetching photo:', err);
    res.status(500).json({ error: 'Failed to fetch photo' });
  }
});

// Update report status (e.g. active → found)
router.patch('/:reportId/status', async (req, res) => {
  try {
    const { reportId } = req.params;
    const { status } = req.body;

    const allowed = ['active', 'found', 'investigating'];
    if (!allowed.includes(status)) {
      return res.status(400).json({ error: 'Invalid status value' });
    }

    const report = await MissingReport.findByIdAndUpdate(
      reportId,
      { status },
      { new: true }
    );

    if (!report) {
      return res.status(404).json({ error: 'Report not found' });
    }

    res.json({
      message: 'Status updated',
      status: report.status,
      id: report._id,
    });
  } catch (err) {
    console.error('Error updating status:', err);
    res.status(500).json({ error: 'Failed to update status' });
  }
});


export default router;
