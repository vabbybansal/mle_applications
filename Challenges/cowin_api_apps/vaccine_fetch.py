import requests
import json
import csv
import datetime
import os
import os.path
from os import path
import logging

# CONSTANTS
URLSTRING_VACCINE_BY_DISTRICT = "https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/findByDistrict?district_id={district_id}&date={datestr}" 
CSV_FILE_NAME = '/Users/vaibhavb/Desktop/repos/dump/mle_applications/Challenges/cowin_api_apps/vaccine_availability.csv'

dir_path = os.path.dirname(os.path.realpath(__file__))
LOG_FILE_NAME = os.path.join(dir_path, '/Users/vaibhavb/Desktop/repos/dump/mle_applications/Challenges/cowin_api_apps/test_log.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def do_logging(message):
    logger.info(message)

DISTRICT_MAP = {
	'New_Delhi' : 140,
	'Gurgaon' : 188	
}
DISTRICT_ID_MAP = {v: k for k, v in DISTRICT_MAP.items()}
headers = {
	'accept': 'application/json', 
	'Accept-Charset': 'UTF-8',
	'Accept-Language': 'hi_IN',
	'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'
}

def get_routine(get_request_url_string, headers):
	r = requests.get(
		get_request_url_string, 
		headers=headers
	)
	return r

def get_vaccine_metrics_by_district(district_id, datestr):
	response_dict_json = get_routine(
			URLSTRING_VACCINE_BY_DISTRICT.format(
					district_id=district_id,
					datestr=datestr
				), 
			headers
		).json()
	
	num_sites = len(response_dict_json['sessions'])
	num_vaccine_type = 0
	vaccine_type_freq = {}

	if num_sites > 0:
		for site in response_dict_json['sessions']:
			vaccine_type = site['vaccine']
			if vaccine_type not in vaccine_type_freq:
				vaccine_type_freq[vaccine_type] = 0
			vaccine_type_freq[vaccine_type] += 1
		num_vaccine_type = len(vaccine_type_freq)

	else: pass

	return [
		datetime.datetime.now(),
		datestr,
		district_id,
		DISTRICT_ID_MAP[district_id],
		num_sites,
		num_vaccine_type,
		json.dumps(vaccine_type_freq)
	]

# # Data Logging
def write_row_vaccine_by_district(data_row):
	if (path.exists(CSV_FILE_NAME) == False):
		with open(CSV_FILE_NAME, 'w', newline='') as file_object:
		    writer = csv.writer(file_object)
		    writer.writerow(["timestamp", "datestr_vaccine_for", "district_id", "district_name", "num_sites", "num_vaccine_types", "vaccine_type_freq_json"])
		    file_object.close()

	with open(CSV_FILE_NAME, 'a', newline='') as file_object:
	    writer = csv.writer(file_object)
	    writer.writerow(data_row)
	    file_object.close()


def main():
	do_logging("started")
	for city_date_tuple in [(DISTRICT_MAP['New_Delhi'], '25-05-2021'), (DISTRICT_MAP['Gurgaon'], '25-05-2021')]:
		data_row = get_vaccine_metrics_by_district(city_date_tuple[0], city_date_tuple[1])
		write_row_vaccine_by_district(data_row)
	do_logging("exiting")
main()

# crontab syntax
# */3 * * * * /Users/vaibhavb/miniconda3/bin/python /Users/vaibhavb/Desktop/repos/mle_applications/Challenges/cowin_api_apps/vaccine_fetch.py
