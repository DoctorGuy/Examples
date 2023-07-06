import time
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import random
import csv
import subprocess
import sys


def copy2clip(txt):
    cmd = 'echo '+txt.strip()+'|clip'
    return subprocess.check_call(cmd, shell=True)


def randomPermutation(arr):
    random.shuffle(arr)
    return arr


text = '0'
seed_words_test = {1: 'north', 2: 'rhythm', 3: 'feature', 4: 'layer', 5: 'coconut', 6: 'ready', 7: 'need', 8: 'final',
                   9: 'camera', 10: 'can', 11: 'early', 12: 'story', 13: 'stable', 14: 'report', 15: 'group', 16: 'depend',
                   17: 'employ', 18: 'problem', 19: 'monitor', 20: 'interest', 21: 'logic', 22: 'sausage', 23: 'toilet', 24: 'pencil'}

EXTENSION_PATH = 'C:/Users/ellio/AppData/Local/Google/Chrome/User Data/Default/Extensions/nkbihfbeogaeaoehlefnkodbefgpgknn/10.28.3_0.crx'

while text == '0':
    opt = webdriver.ChromeOptions()
    opt.add_extension(EXTENSION_PATH)
    driver = webdriver.Chrome(options=opt)
    time.sleep(4)
    driver.switch_to.window(driver.window_handles[1])
    time.sleep(20)
    elem = driver.find_element(
        by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/ul/li[2]/button').click()
    time.sleep(1)
    elem = driver.find_element(
        by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div/button[2]').click()
    time.sleep(1)

    # CREATE KEYWORD STRING
    count = 0
    s = ''
    for e in seed_words_test:
        s += " " + seed_words_test[e]
    s.strip()

    # COPY STRING TO CLIPBOARD:
    copy2clip(s)

    # CLICK ON TEXTFIELD
    time.sleep(1)
    elem = driver.find_element(
        by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[4]/div/div/div[3]/div[1]/div[1]/div/input')
    # PASTE CLIPBOARD TO TEXTFIELD
    elem.send_keys(Keys.CONTROL + 'v')
    time.sleep(1)

    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
           14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    looper = True

    try:
        while looper:
            try:
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[4]/div/div/div[4]')
                count += 1
                s = ''
                arr = randomPermutation(arr)

                for e in arr:
                    s += " " + seed_words_test[e]
                s.strip()

                # COPY STRING TO CLIPBOARD:
                copy2clip(s)
                # click on textfield
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[4]/div/div/div[3]/div[1]/div[1]/div/input')
                # PASTE CLIPBOARD TO TEXTFIELD
                elem.send_keys(Keys.CONTROL + 'v')

                print(s)
                print(count)

            except:
                looper = False
                print("DONE!!!", s)
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[4]/div/button')
                time.sleep(1)
                elem.click()
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[2]/form/div[1]/label/input')
                elem.send_keys('fakepassword123')
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[2]/form/div[2]/label/input')
                elem.send_keys('fakepassword123')
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[2]/form/div[3]/label/input')
                elem.click()
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[2]/form/button')
                elem.click()
                time.sleep(1)
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[2]/button')
                elem.click()
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[2]/button')
                elem.click()
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[1]/div/div[2]/div/div/div/div[2]/button')
                elem.click()
                # Click X
                elem = driver.find_element(
                    by=By.XPATH, value='/html/body/div[2]/div/div/section/div[2]/div/button')
                elem.click()
                elem = driver.find_element(
                    by=By.CLASS_NAME, value="asset-list-item__token-value")
                text = elem.text
    finally:
        # manually cleanup
        time.sleep(3)
        print(text)
        driver.close()
