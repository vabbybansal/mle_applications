import requests
import json
import csv
import datetime
import os
import os.path
from os import path
import logging
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd

# CONSTANTS
URLSTRING_VACCINE_BY_DISTRICT = "https://cdn-api.co-vin.in/api/v2/appointment/sessions/public/calendarByDistrict?district_id={district_id}&date={datestr}" 
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
	
	total_capacity = 0
	num_vaccine_type = 0
	vaccine_type_freq = {}

	for center in response_dict_json['centers']:

		sessions = center['sessions']
		for session in sessions:
			session_capacity = session['available_capacity']
			total_capacity += session_capacity
			if session_capacity > 0:
				vaccine_type = session['vaccine']
				if vaccine_type not in vaccine_type_freq:
					vaccine_type_freq[vaccine_type] = 0
				vaccine_type_freq[vaccine_type] += session_capacity
		num_vaccine_type = len(vaccine_type_freq)

	return [
		datetime.datetime.now(),
		datestr,
		district_id,
		DISTRICT_ID_MAP[district_id],
		total_capacity,
		num_vaccine_type,
		json.dumps(vaccine_type_freq)
	]

# # Data Logging
def write_row_vaccine_by_district(data_row):
	if (path.exists(CSV_FILE_NAME) == False):
		with open(CSV_FILE_NAME, 'w', newline='') as file_object:
		    writer = csv.writer(file_object)
		    writer.writerow(["timestamp", "datestr_vaccine_for", "district_id", "district_name", "total_capacity", "num_vaccine_types", "vaccine_type_freq_json"])
		    file_object.close()

	with open(CSV_FILE_NAME, 'a', newline='') as file_object:
	    writer = csv.writer(file_object)
	    writer.writerow(data_row)
	    file_object.close()

def mailer(email_html_content, subject):

	# Get mail server secrets from environment
	port = int(os.environ.get("COVID_SCRIPT_VAR_PORT"))  # For SSL
	smtp_server = os.environ.get("COVID_SCRIPT_VAR_SMPT_SERVER")
	sender_email = os.environ.get("COVID_SCRIPT_VAR_SENDER_EMAIL")  # Enter your address
	receiver_email = os.environ.get("COVID_SCRIPT_VAR_RECEIVER_EMAIL")  # Enter receiver address
	password = os.environ.get("COVID_SCRIPT_VAR_PASSWORD")

	message = MIMEMultipart("alternative")
	message["Subject"] = subject
	message["From"] = sender_email
	message["To"] = receiver_email

	# Create the plain-text and HTML version of your message
	message.attach(MIMEText(email_html_content, "html"))

	context = ssl.create_default_context()
	with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, message.as_string())


def main():
	do_logging("started")
	today_datestr = (datetime.date.today() + datetime.timedelta(days=0)).strftime("%d-%m-%Y")
	for city_date_tuple in [(DISTRICT_MAP['New_Delhi'], today_datestr), (DISTRICT_MAP['Gurgaon'], today_datestr)]:
		data_row = get_vaccine_metrics_by_district(city_date_tuple[0], city_date_tuple[1])
		write_row_vaccine_by_district(data_row)

		# Custom Alerts - Alert if there is an increment in th Covaxin supply in Gurgaon {since vaccine availability from the API does not mean avialble slots on vowin, so only alert when there is an increase in supply}
		if data_row[3] == 'Gurgaon' and 'COVAXIN' in json.loads(data_row[-1]):
			vaccine_db = pd.read_csv(CSV_FILE_NAME)
			last_vaccine_count = json.loads(vaccine_db[vaccine_db['district_name']=='Gurgaon'].iloc[-1]['vaccine_type_freq_json']).get('COVAXIN') or 0
			new_vaccine_count = (json.loads(data_row[-1]).get('COVAXIN') or 0)

			if new_vaccine_count > last_vaccine_count:
				email_html_content = """\
					<html>
					  <body>
					   	<div>{slots_info}</div> 
					  </body>
					</html>
					""".format(slots_info=data_row[-1])
				subject = "Gurgaon Covaxin Vaccine Slots Available"
				mailer(email_html_content, subject)
	do_logging("exiting")
main()

# crontab syntax
# */3 * * * * EXPORT ENV_VAR="xx"; /Users/vaibhavb/miniconda3/bin/python /Users/vaibhavb/Desktop/repos/mle_applications/Challenges/cowin_api_apps/vaccine_fetch.py
