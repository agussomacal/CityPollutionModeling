import time
from collections import namedtuple
from datetime import datetime
from pathlib import Path

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
        int(zoom), datetime.datetime(*list(map(int, [year, month, day, hour, minute])))


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
    try:
        pollution, station_coordinates = prepare_pollution_data()
        station_coordinates = station_coordinates.loc[
                              (48.808707 <= station_coordinates["lat"]) & (station_coordinates["lat"] <= 48.907608) &
                              (2.246703 <= station_coordinates["long"]) & (station_coordinates["long"] <= 2.430724), :]
        print(f"Number of stations in Paris to consider: {len(station_coordinates)}")
        print(station_coordinates.index)
        station_coordinates.index.name = "Nom_Station"
        station_coordinates.reset_index("Nom_Station", inplace=True)
    except:
        station_coordinates = pd.DataFrame([
            ["Rue Bonaparte", 2.334445, 48.856284],
            ["Place de l'Opéra", 2.332494, 48.870276],
            ["PARIS Centre", 2.350980, 48.859298],
            ["Bld Périphérique Est", 2.412672, 48.838545],
            ["PARIS stade Lenglen", 2.269864, 48.830384]],
            columns=["Nom_Station", "long", "lat"]
        )

    stations_traffic_dir = Path.joinpath(config.data_dir, f"TrafficStationsScreenshots_{screenshot_period}")
    stations_traffic_dir.mkdir(parents=True, exist_ok=True)

    stations_dir = dict()
    for st in station_coordinates.itertuples():
        # To save the ScreenShots in a special Results folder created automatically
        stations_dir[st.Nom_Station] = Path.joinpath(stations_traffic_dir, f"TrafficScreenshots_{st.Nom_Station}")
        stations_dir[st.Nom_Station].mkdir(parents=True, exist_ok=True)
        # take background to substract image afterwards
        do_screenshot(driver, folder=stations_dir[st.Nom_Station], lat=st.lat, long=st.long, zoom=15, traffic=False)

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
                do_screenshot(driver, folder=stations_dir[st.Nom_Station], lat=st.lat, long=st.long, zoom=15)
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
