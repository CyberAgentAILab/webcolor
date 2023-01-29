## Development environment

-   Google Chrome: 108.0.5359.124 
-   ChromeDriver: 108.0.5359.71
-   Lighthouse: 9.6.8

### Install Google Chrome [[link](https://www.google.com/chrome/)]
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
sudo apt -fy install
rm google-chrome-stable_current_amd64.deb
```

### Install ChromeDriver [[link](https://chromedriver.chromium.org/downloads)]
```bash
wget https://chromedriver.storage.googleapis.com/108.0.5359.71/chromedriver_linux64.zip
unzip chromedriver_linux64.zip
sudo mv chromedriver /usr/local/bin/
rm chromedriver_linux64.zip
```

### install Node.js v16.x and Lighthouse [[link](https://github.com/GoogleChrome/lighthouse)]
```bash
curl -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt install -y nodejs
sudo npm install -g lighthouse
```
