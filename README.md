# Smart Surveillance & Communication System

---

## Features

* User & Police authentication
* Missing person reporting
* Image storage in MongoDB (BLOB)
* Police dashboard
* JWT-based security
* REST API architecture
* Scalable MERN backend

---

## Tech Stack

### Frontend

* React + TypeScript
* Tailwind CSS
* Framer Motion

### Backend

* Node.js
* Express.js
* JWT Authentication
* bcrypt
* Multer

### Database

* MongoDB Atlas
* Mongoose ODM
* Binary (BLOB) image storage

---

## User Roles

* **Citizen/User**

  * Register & login
  * Report missing persons
  * Upload photos

* **Police**

  * Secure login
  * View and manage cases
  * Dashboard access


---

## Security

* Hashed passwords (bcrypt)
* JWT-based stateless authentication
* Role-based route protection
* Secure data handling

---

## Image Storage

* Images processed on backend
* Stored directly in **MongoDB as BLOB**
* Linked to corresponding reports
* No dependency on local upload directories

---

## Application Flow

1. User logs in → JWT issued
2. Missing person report submitted with image
3. Backend validates and stores data
4. Police dashboard fetches secured data
5. Real-time UI updates on frontend

---

## Getting Started

### Backend

```bash
cd Backend
npm install
node index.js
```

### Frontend

```bash
cd Frontend
npm install
npm run dev
```

> Add MongoDB URI, port and JWT secrets in `.env` files.

---

## 📂 Folder Structure

```
Frontend/
  └─ src
    ├─ components/
    ├─ context/
    └─ App.tsx

Backend/
  ├─ middleware/
  ├─ models/
  ├─ routes/
  ├─ index.js
  └─ seedLocation.js
```

---
