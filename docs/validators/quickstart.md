# Quickstart


## Dependencies

You must have the following things:

- System with at least 12gb of VRAM
- Python >=3.10
- OpenAI API key
- Anthropic API Key
- Google Gemini API Key
- OpenRouter api key
- Github Token
- Wandb account
- Docker on validator machine
- Docker on a docker server (see below, digital ocean droplet is recommended)
- Optional: Corcel API Key
## Getting started


## Installation

This repository requires python3.11, follow the commands below to install it if you do not already have it.

ONLY RUN THE FOLLOWING COMMANDS IF YOU DO NOT HAVE PYTHON INSTALLED
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

Ensure that your python version is 3.11 before continuing:
```bash
python3 --version
```

If the above doesnt return `python3.11` try using the command `python3.11` instead. If the cmd `python3.11` works, use that in place of every python command below. 

YOU WILL GET SOME ERRORS ABOUT THE PYTHON VERSION, IGNORE THEM.

After ensuring you have python run the following commands:
```bash
git clone https://github.com/brokespace/code
cd code
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --use-deprecated=legacy-resolver -r requirements.txt
python3 -m pip install --use-deprecated=legacy-resolver -e .
python3 -m pip uninstall uvloop # b/c it causes issues with threading/loops
```



#### Setup your dotenv

Copy `.env.example` to `.env` - `cp .env.example .env`. Then edit the `.env` file with the github token you get below

#### Get a Github Token

We require github tokens, to get one follow the instructions [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens), or below.

1. Go to [Github](http://Github.com)
2. Open the top right menu and select `Settings`
3. Go to the bottom left and select `Developer Settings`
4. Go to either `Tokens (classic)` or `Fine-grained tokens`
5. Generate a new token and place it in the .env

#### Get an OpenAI Key

To use OpenAI's services, you need to obtain an API key. Follow the steps below to get your OpenAI API key:

1. Go to the [OpenAI website](https://www.openai.com/).
2. Sign up for an account if you don't already have one, or log in if you do.
3. Navigate to the API section of your account.
4. Generate a new API key.
5. Copy the API key and store it in a secure location.

Once you have your OpenAI API key, add it to your `.env` file like this:

```
OPENAI_API_KEY=<your openai api key>
```

#### Get a Corcel API Key

This step is optional but recommended to reduce validating costs.

Head to [Corcel](https://app.corcel.io) and sign up for an account.

Once you have an account, create an API key and place it in the .env file like this:

```
CORCEL_API_KEY=<your corcel api key>
```

#### Get a Chutes API Key

As a validator you will need to use the Chutes API to validate the miner submissions.

Instructions for getting an API Key can be found [here](https://github.com/rayonlabs/chutes?tab=readme-ov-file#-validators-and-subnet-owners).

Place the api key in the .env file like this:

```
CHUTES_API_KEY=<your chutes api key>
```


#### Get a Claude API Key

Place the api key in the .env file like this:

```
ANTHROPIC_API_KEY=<your anthropic api key>
```

#### Get a Gemini API Key

Place the api key in the .env file like this:

```
GOOGLE_API_KEY=<your gemini api key>
```


#### Get OpenRouter API Key

Place the api key in the .env file like this:

```
OPENROUTER_API_KEY=<your openrouter api key>
```



#### Install Docker (ON VALIDATOR MACHINE)

See instructions [here](https://docs.docker.com/engine/install/)

Ensure that the default user has access to the docker daemon.

```bash
sudo usermod -aG docker $USER
sudo chmod 666 /var/run/docker.sock
```

Update the `/etc/docker/daemon.json` file with the following content:
```bash
{
  "insecure-registries": ["<ip-of-docker-server>:5000"]
}
```

Then restart docker:

```bash
sudo systemctl restart docker
```

#### Setup Docker Server

Setup the docker server to host the miner submissions.

[Docker Server Quickstart](./swe.md)

#### Setup LLM Server

Start the server:

```bash
source .venv/bin/activate
cd coding/finetune/llm
pm2 start --name llm-server.25000 "gunicorn app:app --workers 5 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:25000 --timeout 800"
```

Ensure that the port 25000 is open on your machine and accessable from the Docker server.

Ensure that ufw is enabled on your machine, after doing so you can restrict the port to only be accessable from the Docker server by running the following commands:

```bash 
sudo ufw allow from <docker-server-ip> to any port 25000
sudo ufw deny 25000
sudo ufw reload
```


Test that the port is open by running the following command from the docker server:

```bash
curl <validator-ip>:25000
```

The command should return the response: `{"detail":"Not Found"}`. If it does not, then the port is not open or accessable from the Docker server.

#### Setup IP Addresses

Setup the IP addresses in the .env file like this:

```
DOCKER_HOST_IP=<docker-server-ip>
HOST_IP=<validator-server-ip>
REMOTE_DOCKER_HOST=tcp://<docker-server-ip>:2375
```

#### Setup LLM Auth Key

Setup the LLM auth key in the .env file like this:

```
LLM_AUTH_KEY=<random auth key>
```

#### Increase Ulimit

```bash
ulimit -n unlimited
```

#### Start the validator



```bash
source .venv/bin/activate
python3 scripts/start_validator.py
    --netuid 45
    --subtensor.network <finney/local/test>
    --neuron.device cuda
    --wallet.name <your wallet> # Must be created using the bittensor-cli
    --wallet.hotkey <your hotkey> # Must be created using the bittensor-cli
    --logging.debug # Run in debug mode, alternatively --logging.trace for trace mode
    --axon.port # VERY IMPORTANT: set the port to be one of the open TCP ports on your machine
    --wandb.on True # default is true but you can disable
```


