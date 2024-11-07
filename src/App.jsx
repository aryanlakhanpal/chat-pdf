import React, { useState } from "react";
import "./App.css";
import { IoCodeSlash, IoSend } from "react-icons/io5";
import { BiPlanet } from "react-icons/bi";
import { FaPython } from "react-icons/fa";
import { TbMessageChatbot } from "react-icons/tb";
import { GoogleGenerativeAI } from "@google/generative-ai";
import SyntaxHighlighter from "react-syntax-highlighter/dist/esm/default-highlight";
import { vs2015 } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { useDropzone } from "react-dropzone"; // Import Dropzone

const App = () => {
  const [message, setMessage] = useState("");
  const [isResponseScreen, setisResponseScreen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [files, setFiles] = useState([]); // State to manage uploaded files

  const onDrop = (acceptedFiles) => {
    setFiles([...files, ...acceptedFiles]);
  };

  const hitRequest = () => {
    if (message) {
      generateResponse(message);
    } else {
      alert("You must write somthing... !");
    }
  };

  const generateResponse = async (msg) => {
    if (!msg) return;

    const genAI = new GoogleGenerativeAI(
      "AIzaSyC8eLYUkIxQUT6_ghrwOi7whE0t1eLvvIU"
    );
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
    const result = await model.generateContent(msg);

    const newMessages = [
      ...messages,
      { type: "userMsg", text: msg },
      { type: "responseMsg", text: result.response.text() },
    ];

    setMessages(newMessages); // Append new messages to the existing ones
    setisResponseScreen(true);
    setMessage(""); // Clear the input field after sending the message
  };

  const newChat = () => {
    setisResponseScreen(false);
    setMessages([]); // Clear the messages array
  };

  const checkEnterPress = (e) => {
    if (e.key === "Enter") {
      hitRequest();
    }
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop }); // Dropzone hooks

  return (
    <div className="container w-screen min-h-screen overflow-x-hidden bg-[#0E0E0E] text-white flex">
      {/* Main Content */}
      <div className="flex-grow">
        {isResponseScreen ? (
          <div className="h-[80vh] pb-4 overflow-y-auto overflow-x-hidden">
            <div className="header pt-[25px] flex items-center justify-between w-[100vw] px-[300px]">
              <h2 className="text-2xl">AssistMe</h2>
              <button
                id="newChatBtn"
                className="bg-[#181818] p-[10px] rounded-[30px] cursor-pointer text-[14px] px-[20px]"
                onClick={newChat}
              >
                New Chat
              </button>
            </div>
            <div className="messages">
              {messages?.map((msg, index) => (
                <div key={index} className={msg.type}>
                  <p>{msg.text}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="middle h-[80vh] flex items-center flex-col justify-center">
            <h1 className="text-4xl">
              Doc Interaction Model using Bert<br></br>
              <br></br>What we used ??
            </h1>
            <div className="boxes mt-[30px] flex items-center gap-2">
              <div className="card rounded-lg cursor-pointer transition-all hover:bg-[#201f1f] px-[20px] relative min-h-[20vh] bg-[#181818] p-[10px]">
                <p className="text-[18px]">
                  langchain <br></br>requests
                </p>
              </div>
              <div className="card rounded-lg cursor-pointer transition-all hover:bg-[#201f1f] px-[20px] relative min-h-[20vh] bg-[#181818] p-[10px]">
                <p className="text-[18px]">
                  sentence_transformers <br></br>bs4
                </p>
              </div>
            </div>
          </div>
        )}

        <div className="bottom w-[100%] flex flex-col items-center">
          <div className="inputBox w-[60%] text-[15px] py-[7px] flex items-center bg-[#181818] rounded-[30px]">
            <input
              onKeyUp={checkEnterPress}
              value={message}
              onChange={(e) => {
                setMessage(e.target.value);
              }}
              type="text"
              className="p-[10px] pl-[15px] bg-transparent flex-1 outline-none border-none"
              placeholder="Write your message here..."
              id="messageBox"
            />
            {message === "" ? (
              ""
            ) : (
              <i
                className="text-green-500 text-[20px] mr-5 cursor-pointer"
                onClick={hitRequest}
              >
                <IoSend />
              </i>
            )}
          </div>
          <p className="text-[gray] text-[14px] my-4">
            Made by tech team Aryan lakhanpal, Aryan pandey, Riya Tyagi and
            Mansi Tiwari
          </p>
        </div>
      </div>

      {/* File Drop Area */}
      <div className="w-[25%] h-full bg-[#181818] flex flex-col items-center justify-center p-4">
        <div
          {...getRootProps()}
          className="w-full h-[200px] border-2 border-dashed border-gray-400 flex flex-col items-center justify-center cursor-pointer"
        >
          <input {...getInputProps()} />
          <p>Drag & Drop files here</p>
          <p>or click to select files</p>
        </div>
        {files.length > 0 && (
          <div className="mt-4">
            <h4>Uploaded Files:</h4>
            <ul>
              {files.map((file, index) => (
                <li key={index}>{file.name}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
