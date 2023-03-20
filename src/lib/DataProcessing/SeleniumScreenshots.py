import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.firefox import GeckoDriverManager

import config

CoordsTuple = namedtuple("Coords", "latitude longitude")
center_of_paris = CoordsTuple(latitude=48.8580073, longitude=2.3342828)

window_size = {'width': 1253, 'height': 1253}


def do_screenshot(driver, folder, lat, long, zoom, traffic=True):
    t = time.gmtime()
    try:
        driver.set_window_size(window_size['width'], window_size['height'])
        # url test of traffic in Paris
        url = f'https://www.google.com/maps/@{lat},{long},{zoom}z' + ("/data=!5m2!1e4!1e1" if traffic else "")
        driver.get(url)  # go to google-maps Paris url

        if traffic:
            # Eliminate relief
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                (By.XPATH,
                 "/html/body/div[3]/div[9]/div[23]/div[1]/div[2]/div[5]/div/div/span[5]/div/span[2]/button/span/span"))).click()

        # save screenshot in specified location
        filename = f"Screenshot_{lat}_{long}_{zoom}_{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}" if traffic else f"Background_{lat}_{long}_{zoom}"
        driver.save_screenshot(f'{folder}/{filename}.png')
    except:
        print(f"Aborted screenshot for: {folder} and {t.tm_mday} {t.tm_hour}:{t.tm_min}")


def get_info_from_name(fname):
    name, lat, long, zoom, year, month, day, hour, minute = fname.split("/")[-1].split("_")
    lat, long = list(map(float, [lat, long]))
    minute = minute.split(".")[0]
    year, month, day, hour, minute = list(map(int, [year, month, day, hour, minute]))
    return name, CoordsTuple(latitude=lat, longitude=long), \
        int(zoom), datetime(*list(map(int, [year, month, day, hour, minute])))


def traffic_screenshots_folder(screenshot_period):
    # To save the ScreenShots in a special Results folder created automatically
    selenium_test_dir = Path.joinpath(config.traffic_dir, f"TrafficScreenshots_{screenshot_period}")
    selenium_test_dir.mkdir(parents=True, exist_ok=True)
    return selenium_test_dir


if __name__ == "__main__":
    screenshot_period = 15  # in minutes
    selenium_test_dir = traffic_screenshots_folder(screenshot_period)

    # ---------- ----------- ----------- ----------- #
    #                   Web driver
    # ---------- ----------- ----------- ----------- #
    try:
        # Driver for automatic firefox managing
        options = webdriver.FirefoxOptions()
        options.add_argument("--headless")
        options.add_argument(f'window-size={window_size["width"]}x{window_size["height"]}')
        options.add_argument("disable-gpu")
        driver = webdriver.Firefox(executable_path=GeckoDriverManager().install(), options=options)
    except:
        # Driver for automatic chrome managing
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument(f'window-size={window_size["width"]}x{window_size["height"]}')
        options.add_argument("disable-gpu")
        driver = webdriver.Chrome(options=options, executable_path=ChromeDriverManager().install())

    # first connection to refuse cookies
    # url test of traffic in Paris
    url = f'https://www.google.com/maps/@48.8710381,2.3446359,13z/data=!5m2!1e4!1e1'
    driver.get(url)  # go to google-maps Paris url
    # click on Tout refuser -> cookies
    # https://stackoverflow.com/questions/64032271/handling-accept-cookies-popup-with-selenium-in-python
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/c-wiz/div/div/div/div[2]/div[1]/div[3]/div[1]/div[1]/form[1]/div/div/button'))).click()

    # ---------- ----------- ----------- ----------- #
    #                   Core of code
    # ---------- ----------- ----------- ----------- #
    station_coordinates = pd.DataFrame([
        ['ELYS', 'PA07', 'AUT', 'BONAP', 'OPERA', 'HAUS', 'PA04C', 'PA13',
         'CELES', 'PA18', 'BASCH', 'PA15L', 'PA12', 'BP_EST', 'SOULT'],
        [48.86867978171355, 48.85719963458387, 48.84951967972958, 48.85621636967545, 48.870360489330714,
         48.87329987270092, 48.85943820267798, 48.828608029768915, 48.85261565506112, 48.8916687478872,
         48.82770896097115, 48.830548878814895, 48.83720120744435, 48.83860845472763, 48.83802336734008],
        [2.311806320972728, 2.293299113256733, 2.253411499806541, 2.3343433611339512, 2.3322591802886805,
         2.330387763170951, 2.351109707540651, 2.360279635559221, 2.3601153218686632, 2.346671039638325,
         2.3267480081515686, 2.2696778063113494, 2.393899257058465, 2.4127801276675642, 2.408116146897623]
    ],
        index=["Nom_Station", "lat", "long"]).T

    stations_traffic_dir = Path.joinpath(config.data_dir, f"TrafficStationsScreenshots_{screenshot_period}")
    stations_traffic_dir.mkdir(parents=True, exist_ok=True)

    for st in station_coordinates.itertuples():
        # To save the ScreenShots in a special Results folder created automatically
        # take background to substract image afterwards
        station_traffic_dir = Path.joinpath(selenium_test_dir, st.Nom_Station)
        station_traffic_dir.mkdir(parents=True, exist_ok=True)
        print("Background for", station_traffic_dir)
        do_screenshot(driver, folder=station_traffic_dir, lat=st.lat, long=st.long, zoom=15, traffic=False)

    # # take background to substract image afterwards
    do_screenshot(driver, folder=selenium_test_dir, lat=center_of_paris.latitude, long=center_of_paris.longitude,
                  zoom=13,
                  traffic=False)
    while True:
        if time.gmtime().tm_min % screenshot_period == 0:
            print(f"Taking screenshots: {time.gmtime()}")
            do_screenshot(driver, folder=selenium_test_dir, lat=center_of_paris.latitude,
                          long=center_of_paris.longitude,
                          zoom=13)
            for st in station_coordinates.itertuples():
                station_traffic_dir = Path.joinpath(selenium_test_dir, st.Nom_Station)
                station_traffic_dir.mkdir(parents=True, exist_ok=True)
                do_screenshot(driver, folder=station_traffic_dir, lat=st.lat, long=st.long, zoom=15)
            time.sleep(60)

    # zoom ratios
    # 19 : 1128.497220
    # 18 : 2256.994440
    # 17 : 4513.988880
    # 16 : 9027.977761
    # 15 : 18055.955520
    # 14 : 36111.911040
    # 13 : 72223.822090
    # 12 : 144447.644200
    # 11 : 288895.288400
    # 10 : 577790.576700
    # 9  : 1155581.153000
    # 8  : 2311162.307000
    # 7  : 4622324.614000
    # 6  : 9244649.227000
    # 5  : 18489298.450000
    # 4  : 36978596.910000
    # 3  : 73957193.820000
    # 2  : 147914387.600000
    # 1  : 295828775.300000
    # 0  : 591657550.500000

    # ---------- ----------- ----------- ----------- #
    # Old way of doing in case the code "/data=!5m2!1e4!1e1" changes meaning
    # # click on land icon -> gives satellite view
    # land_icon = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
    #     (By.XPATH, '//*[@id="minimap"]/div/div[2]/button')))
    # land_icon.click()
    # # click again on land icon -> goes back to streets view but opens also the panel for further options
    # land_icon.click()
    # # more layers option
    # WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
    #     (By.XPATH, '/html/body/div[3]/div[9]/div[23]/div[7]/div/div/div/ul/li[5]/button/div'))).click()
    # # cancel labels: 2022 12 06 google stopped allowing unclicking the labels
    # # WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
    # #     (By.XPATH, '/html/body/div[3]/div[9]/div[24]/div/div/div/div[3]/ul/li[2]/button/div[1]'))).click()
    # # closes the options panel
    # WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
    #     (By.XPATH, '/html/body/div[3]/div[9]/div[24]/div/div/div/div[1]/header/button'))).click()
    #
    # # click in the center of the screen
    # action = webdriver.common.action_chains.ActionChains(driver).move_by_offset(original_size['width'] // 2,
    #                                                                             height // 2).click().perform()
    #
    # # take lat long coordinates
    # coords = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
    #     (By.XPATH, '/html/body/div[3]/div[9]/div[23]/div[1]/div[2]/div[2]/div/div[2]/div[3]/button'))).text
    #
    # coords = list(map(float, coords.split(",")))
    # print(coords)
    #
    # # save screenshot in specified location
    # driver.save_screenshot(f'{selenium_test_dir}/Screenshot.png')
    # driver.quit()
