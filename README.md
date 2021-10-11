# Babur (Slack Bot)

## Who is Babur?

Babur is ml-based real-time Slack Bot. It categorizes the messages sent to the channels you specify in Slack and returns the answers you specify.

![babur-bot](https://user-images.githubusercontent.com/52164941/136689117-0e5a63f4-571b-4e82-b04d-2ae286df831c.png)

## 🔨 How does it work?

### Train model section
- create environment:
```
 conda env create -f=environment.yaml
 conda activate babur
```

- Firstly you should have labeled data to train a model.
-------------

<details>
<summary>If you don't have data</summary> 
    You can listen your slack channels and write them to the google sheets or excel.  
      <ol>
      <li> We have <a href="https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/slack_message_helper.py"> <p>slack_message_helper</p> </a> class.</li>
      <li>In this class we get trained model, listen message and analyse, return answer, write to google sheet all of data. Now you dont have trained model yet. So you should use just listen messages and write to google sheet. After you can label messages and train a model with them. </li>
    </ol>
</details>

------------

- In model.py we get our messages from Google Bigquery.(We connect google sheet to google bigquery.) If your data on local csv file or others, you can use [load_data_from_csv_file](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/model/model.py#L36) or you can write your own.
- You should explain your google application credential to use google bigquery.
  > export GOOGLE_APPLICATION_CREDENTIALS=path-to-your-key-json-file.json 
- In the rest, we create the best model by trying many algorithms.


### "slack_message_helper" class 


First you need to fill [there](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/.env.example) according yourself.
  - LISTENING_CHANNELS: We just listen, not answer yet. (So if you collect data first, you can get the IDs of channels you will listen to here.)
  - TARGET_CHANNELS: We listen and answer to this channels.

We have [slack_message_helper](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/slack_message_helper.py) class. 
| Function      | Description |
| ----------- | ----------- |
| get_response      | Get the message, analyse them via *analyze_message* function, return response which you determine.       |
| analyze_message   | Get the message, applies to preprocess steps, return models prediction.        |
| get_sheet_cover   | Write message metada which you specify to google sheet.         |


### Set up slack app to answer


- You need to have app in slack. Check [here](https://slack.com/intl/en-cy/help/articles/115005265703-Create-a-bot-for-your-workspace).
- If you have, open your slack app [page](https://api.slack.com/apps)
- Open Event Subscription page and give your domain with "/slack/events" endpoint.(If you run on local, you can use ngrok.)
- Make sure that you have the necessary permissions.


### App side


- We're on [bot.py](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/bot.py)
- You should give your slack_api_token and signing secret.
- We have a function app.event("message") here. This listens every message from channels which bot added in.
- In the rest of function we return answers with using slack_message_helper class. We dont return answers to our team members so we add them to no_reply_team_members.
- Our web application will be run on the 5000 port. Remember, we gave our endpoint to slack. Slacks send each message payload to us via a POST request.
- We use gunicorn to serve our application. We can use a shell script to run the app. Create a file called run.sh with the following content.
   > sh run.sh  

- We build an Nginx server to manage our application requests. Start with creating a folder named nginx that contains two files *nginx.conf* and *project.conf*.


### Dockerize


- Under nginx folder we got [Dockerfile](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/nginx/Dockerfile) . This file will create an image with our Nginx configurations.
- Create docker image for bot [here](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/Dockerfile)
- To stand up our container create [docker-compose.yml](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/docker-compose.yml)
- We create [train.sh](https://github.com/Bhasfe/babur/blob/8cc143dee0a037742080a2509320b89c3d759b87/train.sh) to schedule training model with cron job.

- Finally build our images and start!
```
    docker-compose build --no-cache
    docker-compose up -d
```


### 💥 What's going on now?

![1_rI3AE3osXJT0OjvThe0vEQ](https://user-images.githubusercontent.com/52164941/136689615-9532f856-61a8-4a6f-b11b-45a7ce6d8dd7.png)


- Messages come from slack to our app.
- If we listen this message channel:
   - Get the message
   - Analyse message with the best model
   - Predict message content and return answer according to content.
   - We also save the incoming message to excel. We can label the messages in excel and use them to train our model.
   
<img width="433" alt="Screen Shot 2021-10-11 at 17 58 15" src="https://user-images.githubusercontent.com/52164941/136813628-e2977728-9787-4d32-9125-69485839676c.png">

### Licence
Distributed under the MIT License. See [LICENSE](https://github.com/Bhasfe/babur/blob/main/LICENSE) for more information.


Please give your feedback ❤️ 
Happy coding!
