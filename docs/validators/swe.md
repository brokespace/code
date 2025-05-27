# SWE Start


## Remote Server Setup

You should use a separate server from the one you run the validator on for this. This is to ensure security and avoid any potential issues. I recommend using a digital ocean droplet. A small one is fine, maybe 2-4gb of ram. 

### Setup Docker

Install docker: https://docs.docker.com/engine/install/ubuntu/

Next setup https://docs.docker.com/engine/daemon/remote-access/#configuring-remote-access-with-daemonjson with 0.0.0.0:2375 - Do so by running the following commands:

```bash
sudo systemctl edit docker.service
```

Add the following to the file at the line where it opens:
```bash
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// -H tcp://0.0.0.0:2375
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker.service
```

### Get Base Image

```bash
docker pull brokespace/swe-server:latest
docker pull registry:2
```

### Configure UFW

```bash
sudo ufw disable
```


### IPTables 
```bash
sudo apt-get install iptables-persistent
```

The order of the rules is important. Run the following commands to setup the rules:


Let docker manage the iptables rules update file `/etc/docker/daemon.json` with the following content:
```bash
{
  "iptables": true,
  "insecure-registries": ["<ip-of-docker-server>:5000"]
}
```

Then restart docker:
```bash
sudo systemctl restart docker
```

```bash
sudo apt install ipset
```

Create a file in `/etc/cron.monthly/dockerio` with the following content:

MAKE SURE YOU SET THE IP OF THE SERVER YOU ARE RUNNING THE VALIDATOR ON IN THE IPTABLES RULES BELOW. 

ONLY RUN THIS IF YOU ARE USING A DIGITAL OCEAN DROPLET. ELSE SEE BELOW FOR ALTERNATIVE METHODS.

```bash
#!/bin/bash
sudo iptables -F
sudo iptables -t nat -F
sudo iptables -t mangle -F
sudo iptables -t raw -F

# Define the IP set name
IPSET_NAME="dockerio"

# Check if the IP set exists; create it if it doesn't
if ! ipset list $IPSET_NAME &>/dev/null; then
    sudo ipset create $IPSET_NAME hash:ip
fi

# Clear existing IPs in the set
sudo ipset flush $IPSET_NAME

# Resolve required domains and add to ipset
for domain in registry-1.docker.io auth.docker.io cdn.docker.io; do
    for ip in $(dig +short $domain); do
        sudo ipset add $IPSET_NAME $ip
    done
done

# Add iptables rules for the IP set
sudo iptables -A OUTPUT -m set --match-set $IPSET_NAME dst -p tcp --dport 443 -j ACCEPT
sudo iptables -A OUTPUT -m set --match-set $IPSET_NAME dst -p tcp --dport 80 -j ACCEPT

# Restart Docker to apply changes
sudo systemctl restart docker

sudo iptables -N DOCKER-USER
sudo iptables -A DOCKER-USER -p tcp --dport 3000 -j ACCEPT
sudo iptables -I DOCKER-USER 1 -p tcp --dport 3000 -j ACCEPT
sudo iptables -I DOCKER-USER 1 -p tcp --dport 25000 -j ACCEPT
# Allow forwarding from your host interface to the Docker bridge
sudo iptables -A FORWARD -p tcp -d 172.17.0.0/16 --dport 3000 -j ACCEPT
sudo iptables -A FORWARD -p tcp -s 172.17.0.0/16 --sport 3000 -j ACCEPT
sudo iptables -A INPUT -p tcp -s <ip-of-server-you-are-running-the-validator-on> --dport 2375 -j ACCEPT
sudo iptables -A INPUT -p tcp -s <ip-of-server-you-are-running-the-validator-on> --dport 5000 -j ACCEPT
sudo iptables -A OUTPUT -p tcp -s <ip-of-server-you-are-running-the-validator-on> --dport 2375 -j ACCEPT
sudo iptables -A OUTPUT -p tcp -s <ip-of-server-you-are-running-the-validator-on> --dport 5000 -j ACCEPT
sudo iptables -I INPUT 1  -p tcp -s <ip-of-server-you-are-running-the-validator-on> --dport 5000 -m conntrack --ctstate NEW,ESTABLISHED -j ACCEPT
sudo iptables -I OUTPUT 1 -p tcp -d <ip-of-server-you-are-running-the-validator-on> --sport 5000 -m conntrack --ctstate ESTABLISHED     -j ACCEPT

# 1. Masquerade traffic that leaves the docker-bridge
sudo iptables -t nat -A POSTROUTING \
       -s 172.17.0.0/16 ! -d 172.17.0.0/16 -j MASQUERADE

# 2. Allow containers to open NEW connections to the validator on 25000
sudo iptables -I FORWARD 1 \
       -p tcp -s 172.17.0.0/16 -d <ip-of-server-you-are-running-the-validator-on> --dport 25000 \
       -m conntrack --ctstate NEW,ESTABLISHED -j ACCEPT

sudo iptables -A INPUT -p tcp --dport 2375 -j DROP
sudo iptables -I OUTPUT 1 -p tcp --dport 25000 -j ACCEPT
sudo iptables -A INPUT -p tcp --sport 25000 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --sport 25000 -j ACCEPT

sudo iptables -A OUTPUT -p udp --dport 53 -j ACCEPT
sudo iptables -A INPUT -p udp --sport 53 -j ACCEPT

sudo iptables -A INPUT -p tcp --sport 443 -j ACCEPT
sudo iptables -I OUTPUT 1 -p tcp --dport 3000 -j ACCEPT
sudo iptables -A OUTPUT -p tcp --dport 3000 -j ACCEPT
sudo iptables -I INPUT 1 -p tcp --dport 3000 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 3000 -j ACCEPT

sudo iptables -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
sudo iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT


# Allow outgoing SSH traffic (port 22)
sudo iptables -A OUTPUT -p tcp --dport 22 -j ACCEPT


# sudo iptables -I OUTPUT 1 -p tcp --dport 25000 -j ACCEPT


# Allow incoming SSH traffic (port 22)
sudo iptables -A INPUT -p tcp --sport 22 -j ACCEPT
sudo iptables -A OUTPUT -j DROP
sudo iptables -A DOCKER-USER -j DROP
sudo iptables -A INPUT -p tcp --dport 2375 -j DROP
sudo iptables -A INPUT -p tcp --dport 5000 -j DROP

sudo iptables-save | sudo tee /etc/iptables/rules.v4
sudo systemctl restart docker

```

Ensure the file is executable:
```bash
sudo chmod +x /etc/cron.monthly/dockerio
```

Run it now:

```bash
sudo /etc/cron.monthly/dockerio
```

If you are not using a digital ocean droplet, you can use the following alternative methods (This has yet to be tested): 

```bash
# Set default policies
sudo ufw default deny incoming
sudo ufw default deny outgoing

# Allow necessary inbound traffic
sudo ufw allow 22 comment 'Allow SSH access'
sudo ufw allow from <ip-of-server-you-are-running-the-validator-on> proto tcp to any port 2375 comment 'Allow access to docker daemon'
sudo ufw allow from <ip-of-server-you-are-running-the-validator-on> proto tcp to any port 5000 comment 'Allow access to docker registry'

# Allow outbound traffic only to the validator server on port 25000
sudo ufw allow out to <ip-of-server-you-are-running-the-validator-on> port 25000 proto tcp comment 'Allow outbound to validator server on port 25000'

```





## Testing Docker Remote Access

From the server you are running the validator on - NOT THE ONE YOU RAN THE ABOVE COMMANDS ON - run the following command:

```bash
curl <docker-server-ip>:2375
```

it should return `{"message":"page not found"}`

Next to test further run from the validator server:

```bash
DOCKER_HOST=tcp://<docker-server-ip>:2375 docker run --rm brokespace/swe-server:latest bash -c "sleep 600"
```

While that command is running you should be able to go onto the docker server and see the container running with the following command:

```bash
docker ps
```