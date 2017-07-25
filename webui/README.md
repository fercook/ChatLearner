# Web User Interface

Python is the number one programming language used for machine learning, while Java is still the most popular language for web development. The purpose of this Web UI is to demonstrate how to present the result of a trained model based on Python and TensorFlow in a Java environment. A SOAP-based web service is generated to meet this need. The steps to create the running environment in a Windows system are described below. Similar procedures can be followed in Linux, but have not been tried.

The description below assumes that you are configuring DNS entry papayachat.net for your machine, and running the web service server and client on the same machine. In order to make the DNS entry papayachat.net work, edit the hosts file (located at C:\Windows\System32\drivers\etc in a normal installation) and add the following line:
    127.0.0.1  		papayachat.net

## Server

Tornado web server (version 4.5.1) and Tornado-webservices (for SOAP web services: https://github.com/rancavil/tornado-webservices) are employed to create the web service server. The Tornado web server is installed while the source code of the Tornado-webservices package is included.

```bash
pip install tornado
```

Do not install the Tornado-webservices package as one of the file (webservices.py) needs to be modified to allow a parameter to be passed into the web service.


When everything is ready, just run the commands below to bring up the web service. You should then be able to see the WDSL file here: 
http://papayachat.net:8080/ChatService?wsdl (or http://localhost:8080/ChatService?wsdl).

```bash
cd webui
cd server
python chatservice.py
```

## chatClient

This Java client is tested with Java 1.7 and Tomcat 7.0. You can try later versions of them if you prefer. Following the steps below to prepare the Java client:

1. Install Java 1.7.
2. Install Tomcat 7.0. Choose port 80 (at least not port 8080 if you are running the python web service on the same machine, which is on port 8080).
3. Copy the whole folder chatClient to C drive as C:\chatClient. 
4. Make changes to the server.xml of the Tomcat installation to add the chatClient web application (copy the Host portion for chatClient into server.xml). You need change the name of the host if you are using a different domain.
6. Make changes to C:\chatClient\ROOT\ajax\getChatReply.jsp the service address if you are using a different domain.
7. Restart Tomcat.

And now you are ready to try it: http://papayachat.net, if you are using this domain.