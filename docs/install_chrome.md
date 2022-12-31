## Development environment

-   Google Chrome: 108.0.5359.124 
-   ChromeDriver: 108.0.5359.71

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
