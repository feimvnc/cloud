This guide can be used to install VMware, RHEL 9.5, Kubernetes v1.31 on local MacBook Pro.

```
Personal computer, software, tools, guide used:
MacBook Pro 2019, 2.3 GHz 8-Core Intel Core i9, Sonoma 14.6.1, 32GB RAM

VMware Fusion 12.2.5 (20904517), user registration is required for download and install for individual learning.
If you have it already, click menu VMware Fusion, check for update, update and install new version.

Download RHEL 9.5 DVD iso, user registration is required for individual use and learning.
Keep user id and password, which will be used on install screen below.
https://developers.redhat.com/products/rhel/download#rhelforsap896
Go to bottom on the page to see section All Downloads, 
Red Hat Enterprise Linux 9.5
Select x_86_64, DVD iso, Release date Nov 12, 2024, Download (10.99 GB)

Kubernetes on RHEL installation guide (followed most steps)
https://infotechys.com/install-a-kubernetes-cluster-on-rhel-9/

Official kubernetes documentation (for learning)
https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/create-cluster-kubeadm/

Visual Studio Code (for ssh terminal remote connection to run commands)
Instructions to install VSCode on MacBook 
https://code.visualstudio.com/docs/setup/mac

kubectl completion (for easy run commands after installation)
https://kubernetes.io/docs/reference/kubectl/generated/kubectl_completion/

kubectl cheatsheet (for common kubectl commands)
https://kubernetes.io/docs/reference/kubectl/quick-reference/


#### 1, Create new VM and install RHEL 9.5 as kubernetes node  (using VMware Fusion)

Open VMWare Fusion, click new, drag rhel-9.5-x86_64-dvd.iso onto popup window
Continue, select Linux, Red Hat Enterprise Linux 8 64-bit (latest option)
Specify the boot firmware: Legacy BIOS
Click Customize Settings, change Save As to: Red Hat Enterprise Linux 8 64-bit-w211 (worker node w211)
Finish

# Important step, configure installation disk on Settings window
click wrench icon, open settings
click Hard Disk (NVMe), Disk size: 20.00 GB
Advanced options: select Pre-allocate disk space (To avoid No disk configured message later), see "Re-configuring disk ...".
Apply

# configure processor and memory on Settings window
click Processors & memory
Processors: 4 processor cores (kubernetes requires minimum 2 core, 2 GB RAM, here it is doubled)
Memory: 4096 MB
Press tab key
close Settings popup window

Note: press Control+Command to display cursor on screen if cursor disappeares
# On virtual machine window, click Play triangle button to start installation
Select
Test this media & install Red Hat Enterprise Linux 9.5 (words in white), Enter
Watch progress on screen

WELCOME TO RED HAT ENTERPRISE LINUX9.5 screen
English, English(United States) (for easy commands later), Continue

INSTALLATION SUMMARY screen
Root Password, click, enter Root Password, Confirm, Done twice (top left)
User Creation, click, Full name, User name, Password, Confirm password, Done twice (top left)

SYSTEM, Installation Destination, Automatic partitioning selected (from above disk configuration)
click, Local Standard Disks, VMware Virtual NVMe Disk, 20 GiB free, Done twice
Automatic partitioning selected (is configured)

SOFTWARE (need to connect to Red Hat registration)
Connect to Red Hat (click to open)
Account (selected), User name, Password (Red Hat user registration data)
Uncheck Connect to Red Hat Insights
Click Register (Registering...), "The system is registered", Done (top left)

SOFTWARE
Software Selection
Server (check), Done (top left)

Begin Installation (button is enabled), click to install
"Downloading 624 RPMs, ..."
After 5-10 minutes, Complete!, Reboot System (click at bottom right)
Also verify VM name on top bar has "*-w211" (for worker node)

# Login terminal (with user created earlier)
hostname; id; uptime # check status
ip a # check ip address to use below ssh

# Follow above guide, https://infotechys.com/install-a-kubernetes-cluster-on-rhel-9/
VMware terminal not friendly to copy/paste commands, use VScode to do the work

Command+SpaceBar, type "terminal" and enter, enter "code" 
Remote Explorer (left), click "+", 
ssh user@0.0.0.0  # replace with above host ip address 
/Users/<your_mac_user>/.ssh/config  

# select this option to save ssh config
A new host entry is added to the left
Right click, Connect to Host in Current Window
Terminal, click New Terminal, enter password in pop-up window
Prompt at bottom panel, [xxx@localhost ~]$  # logged in now
Command+B (close left panel)

ip -a # verify host ip and name 
hostname

# add new user created to sudoers file 
(otherwise user is not in the sudoers file.  This incident will be reported.)
su  # enter password to update below file
sudovi -f /etc/sudoers  # add new user below admin user
escape, :set number  # show line numbers
:101  # go to line 101
escape, i, # enter below
<replace_user_name> ALL=(ALL)  ALL
:wq  # write and quit
exit

# update hostname and add control plane host
sudo vim /etc/hosts

# Kubernetes Cluster
192.168.1.26 	cp201  # Replace with your actual hostname and IP address
192.168.1.27 	wk211   # Replace with your actual hostname and IP address
192.168.1.28 	wk212   # Replace with your actual hostname and IP address

# ping host to validate connectivity
ping cp201
ping wk211

# install kernel headers
sudo dnf install kernel-devel-$(uname -r)  # enter y (3 times), Complete!

# add load kernel modules (on each node)
sudo modprobe br_netfilter
sudo modprobe ip_vs
sudo modprobe ip_vs_rr
sudo modprobe ip_vs_wrr
sudo modprobe ip_vs_sh
sudo modprobe overlay

# create a configuration file (as the root user on each node) to ensure these modules load at system boot:
su

cat > /etc/modules-load.d/kubernetes.conf << EOF
br_netfilter
ip_vs
ip_vs_rr
ip_vs_wrr
ip_vs_sh
overlay
EOF

# Configure Sysctl
su  # (root)

cat > /etc/sysctl.d/kubernetes.conf << EOF
net.ipv4.ip_forward = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.bridge.bridge-nf-call-iptables = 1
EOF

# Run the following command to apply the changes
su # (root)

sysctl --system

# Disabling Swap (k8s requires)
sudo swapoff -a  # disabl swap on each server
sed -e '/swap/s/^/#/g' -i /etc/fstab    # turn off all swap devices

cat /etc/fstab  # check and validate below, # added
#/dev/mapper/vg00-swap   none                    swap    defaults        0 0

# Install Containerd, container runtime responsible for managing and executing containers, Docker CE is the free version of Docker

sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo

# Update Package Cache
sudo dnf makecache  

# install the containerd.io package
sudo dnf -y install containerd.io

# Configure Containerd
# The configuration file for Containerd is located at /etc/containerd/config.toml
cat /etc/containerd/config.toml

# Run the following command to build out the containerd configuration file
sudo sh -c "containerd config default > /etc/containerd/config.toml" ; cat /etc/containerd/config.toml

# set the SystemdCgroup variable to true (SystemdCgroup = true)
sudo vim /etc/containerd/config.toml
:set number
:139
# set below to true
SystemdCgroup = true
: wq

# run the following command to ensure the containerd.service starts up and is enabled to autostart on boot up

sudo systemctl enable --now containerd.service
sudo systemctl reboot
# check status
sudo systemctl status containerd.service
# ... Active: active (running) ...

# allow specific ports used by Kubernetes components through the firewall,
sudo firewall-cmd --zone=public --permanent --add-port=6443/tcp
sudo firewall-cmd --zone=public --permanent --add-port=2379-2380/tcp
sudo firewall-cmd --zone=public --permanent --add-port=10250/tcp
sudo firewall-cmd --zone=public --permanent --add-port=10251/tcp
sudo firewall-cmd --zone=public --permanent --add-port=10252/tcp
sudo firewall-cmd --zone=public --permanent --add-port=10255/tcp
sudo firewall-cmd --zone=public --permanent --add-port=5473/tcp

# reload the firewall to apply the changes
sudo firewall-cmd --reload

# Port(s)	Description
6443	Kubernetes API server
2379-2380	etcd server client API
10250	Kubelet API
10251	kube-scheduler
10252	kube-controller-manager
10255	Read-only Kubelet API
5473	ClusterControlPlaneConfig API

# Install Kubernetes Components
# Add Kubernetes Repository, as root user
su 

cat <<EOF | sudo tee /etc/yum.repos.d/kubernetes.repo
[kubernetes]
name=Kubernetes
baseurl=https://pkgs.k8s.io/core:/stable:/v1.29/rpm/
enabled=1
gpgcheck=1
gpgkey=https://pkgs.k8s.io/core:/stable:/v1.29/rpm/repodata/repomd.xml.key
exclude=kubelet kubeadm kubectl cri-tools kubernetes-cni
EOF

# Install Kubernetes Packages (kubelet, kubeadm, and kubectl)
# The --disableexcludes=kubernetes flag ensures that packages from the Kubernetes repository are not excluded during installation.

dnf makecache; dnf install -y kubelet kubeadm kubectl --disableexcludes=kubernetes

# Start and Enable kubelet Service

systemctl enable --now kubelet.service

# Control plane and worker node setup complete
NOTE: Up until this point of the installation process, we’ve installed and configured Kubernetes components on all nodes. From this point onward, we will focus on the master node.


#### 2, Initializing Kubernetes Control Plane (on control plane node)

# Initializing Kubernetes Control Plane, pull the necessary container images (apiserver, controller, scheduler, proxy, coredns, etcd)

sudo kubeadm config images pull

# The --pod-network-cidr flag specifies the range of IP addresses for the pod network. Adjust the CIDR according to your network configuration if needed.

sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# write down below output to run on worker node later, for example
kubeadm join 172.x.x.x:6443 --token ydhgxh.pvlspwcxxxxxxx \
        --discovery-token-ca-cert-hash sha256:731581f7ff9257e81892411ca98187d40684aaaabbbbcccc000011112222 

# recreate control plane cluster, use below
# sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --ignore-preflight-errors=all

# Set Up kubeconfig File

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Deploy Pod Network, required
# To enable networking between pods across the cluster, deploy a pod network.

kubectl create -f https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/tigera-operator.yaml

# download the custom Calico resources manifest
curl -O https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/custom-resources.yaml

# or wget
#wget https://raw.githubusercontent.com/projectcalico/calico/v3.26.1/manifests/custom-resources.yaml

# Adjust the CIDR setting in the custom resources file (replace default values)
sed -i 's/cidr: 192\.168\.0\.0\/16/cidr: 10.244.0.0\/16/g' custom-resources.yaml 

# create the Calico custom resources (in k8s, required)

kubectl create -f custom-resources.yaml


#### 3, Join Worker Nodes

# Get Join Command on Master Node
sudo kubeadm token create --print-join-command

# sample output, kubeadm join x.x.x.x:6443 --token x.x --discovery-token-ca-cert-hash sha256:x 

# Run Join Command on Worker Nodes
sudo kubeadm join <MASTER_IP>:<MASTER_PORT> --token <TOKEN> --discovery-token-ca-cert-hash <DISCOVERY_TOKEN_CA_CERT_HASH>

# or add --ignore-preflight-errors
# sudo kubeadm join x.x.x.x:6443 --token x.x --discovery-token-ca-cert-hash sha256:x --ignore-preflight-errors=all


#### Operations 

## check cluster status
# on control plan
kubectl get nodes   # check cluster status
kubectl get pods -A  # get pods from all namespaces

## remove work node from cluster, if pods managed by a DaemonSet
alias k=kubectl 
kubectl cordon <replace-node-name>  # delete vm 
kubectl drain <node-name> --force --ignore-daemonsets  # unschedule node
kubectl delete node <node-name>

## watch process, every one second
watch -n 1 kubectl get nodes


#### Test pod deployment using nginx, nodePort, and curl
## Below works on local setup on VMware using RHEL9.5

## create a deployment, a service using nodePort

# create a deployment using below file content
# cat nginx-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        ports:
        - containerPort: 80

kubectl apply -f nginx-deployment.yaml
kubectl get deployments  # show Reasy 3/3
kubectl get pods   # show Running


# create a service using below file content
# cat nginx-service-nodePort.yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
      nodePort: 30080
  type: NodePort

kubectl apply -f nginx-service-nodePort.yaml
kubectl get services 
kubectl get nodes -o wide  # write down worker node ip where pods are running
kubectl get pods -o wide  # check pods status 
kubectl get pod -o yaml | grep ip   # write down worker node ip for curl

# use worker node IP and port 30080 to curl app endpoint
# curl <worker_ip>:<nodePort_port>

curl 172.16.x.x:30080   # replace with your own node ip address
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>

#### end of creation of kubernetes VM, control plane, worker node, deployment, and testing.


#### Extra setup and learning

## Enable Bash Completion

#Install the bash-completion package: 
sudo yum install bash-completion

#Source the kubectl completion script: 
source <(kubectl completion bash)

#Add to your Bash configuration  (~/.bashrc or /etc/bashrc) to persist the changes:
echo "source <(kubectl completion bash)" >> ~/.bashrc

# set alias k=kubectl
echo "alias k=kubectl" >> ~/.bashrc

## config vim
echo "set expandtab shiftwidth=2 tabstop=2 softtabstop=2" > ~/.vimrc 

#" Or the short form
#echo "set et sw=2 ts=2 sts=2" > ~/.vimrc

## check kubernetes version 
k version

## review cluster manifest files
/etc/kubernetes



```