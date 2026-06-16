
##  Getting Started

Follow these steps to set up and run the development server.

### 1. Prerequisites

Ensure you have the following installed on your system:

* **Node.js** (version $\ge 18$)
* **npm** (comes with Node) or **yarn**
* Any modern web browser

## 2. Install Dependencies

Navigate into the `Frontend` directory and run the installation command:

```bash
npm install
# or
yarn install
```

## 3.Setting up environment variable
Create a file named .env at the root of the Frontend directory and add the backend URL.
Global Base Url can be found on --- https://route-ofx8.onrender.com/
```bash
VITE_BACKEND_URL=http://localhost:8000
VITE_GLOBAL_BASE_URL= Our global base url  -- get from this route-- https://route-ofx8.onrender.com/
```

## 4. Run the Development Server
Start the application in development mode:
```bash
npm run dev
```
The application will typically start at: http://localhost:5173/

## 5. Build for Production
To create a production-ready static build, run:
```bash
npm run build
```
The optimized output files will be generated in the dist/ directory.

##  Project Structure
The key directories and files in the frontend are:

```bash
Frontend/
│
├── dist/                     # Production build output (generated after npm run build)
├── node_modules/             # Installed dependencies
│
├── public/                   # Static public assets (e.g., favicon)
│   └── index.html
│
├── src/
│   ├── assets/               # Icons, static images, etc.
│   │
│   ├── Components/           # Reusable UI components
│   │   ├── Add_btn.tsx
│   │   ├── ChatWindow.tsx        # Handles chat and Q&A interactions
│   │   ├── CreateSessionForm.tsx
│   │   ├── DarkModeToggle.tsx
│   │   ├── ImageUploader.tsx     # Handles image upload to the backend
│   │   ├── MainSection.tsx       # Primary view area for displaying results
│   │   ├── Sessions_btn.tsx
│   │   ├── SessionsList.tsx
│   │   └── useTypedMessage.tsx   # Custom hook for animated message typing
│   │
│   ├── Pages/
│   │   └── Home.tsx          # Main landing page/application layout
│   │
│   ├── Store/
│   │   └── store.ts          # Zustand global state management
│   │
│   ├── App.css
│   ├── App.tsx               # Main application component
│   ├── index.css
│   ├── main.tsx              # Entry point for the React application
│   └── ThemeContext.tsx      # React Context for managing light/dark theme
│
├── .env                      # Environment variables configuration
├── vite.config.ts            # Vite configuration file
└── package.json              # Project dependencies and scripts
```


## Frontend Workflow

The application is designed to guide the user through the following process:

Create a Session: Start by creating a new session using a unique name.

Upload an Image: Upload an image within the newly created session.

Choose an Operation: Select a specific model operation from the Hamberger sign in the message input field (e.g.,"Generate Caption", "Generate grounding", "Visual Question Answering" ).

View Results: The processed image (if applicable) and the structured JSON output from the backend are displayed in the chat section.

Continue Interaction: Users can continue chatting or performing more operations on the same uploaded image within the current session.
