from typing import List
from pydantic import BaseModel, ValidationError
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Optional
import uuid 


def compute_lognormal_parameters(mean: float, cv: float):
    """
    Calculate mu and sigma for lognormal distribution given mean and cv.
    Adapted from: https://www.johndcook.com/blog/2022/02/24/find-log-normal-parameters/

    """
    variance = (mean * cv) ** 2
    sigma2 = np.log(1 + variance / (mean**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - sigma2 / 2
    return (mu.item(), sigma.item())

class UUIDGenerator:
    """
    Helper class to generate UUIDs with a specified length of the UUID string.
    The UUID is generated using the uuid4 method and then truncated to the specified length.
    Optional to add prefix and suffix to the generated UUID, which will add to the total length.
    """

    def __init__(self, id_length: int = 10):
        self.id_length = id_length

    def generate_id(
        self, prefix: Optional[str] = "", suffix: Optional[str] = ""
    ) -> str:
        id = prefix + str(uuid.uuid4().hex)[: self.id_length] + suffix
        return id

class GroupProfiles(BaseModel):
    name: List[str]
    txn_mean_low: List[float]
    txn_mean_high: List[float]
    txn_cv_low: List[float]
    txn_cv_high: List[float]
    txn_lambda: List[float]


class Customer:
    def __init__(self, profile: dict):
        self.profile = profile
        self._date_format = "%Y-%m-%d"
        self._timestamp_format = "%Y-%m-%d %H:%M:%S"
        self._big_ticket_proba = 0.005
        self._big_ticket_multiplier = 10  # Adjust the multiplier as needed
        self._uuid_generator = UUIDGenerator(id_length=10)

    def generate_txn_value(self):
        """
        Generate a single transaction value for a customer based on their profile
        In addition, there is a fixed probability (0.5%) of a big-ticket item
        """

        if (
            np.random.rand() < self._big_ticket_proba
        ):  # rand() generates from uniform [0,1), thus there is 0.5% chance that the value is less than 0.005

            # Generate a big-ticket item
            big_ticket_mean = self.profile["txn_mean"] * self._big_ticket_multiplier
            big_ticket_sigma = self.profile["txn_sigma"]
            txn_value = round(
                np.random.lognormal(
                    mean=np.log(big_ticket_mean),
                    sigma=big_ticket_sigma,
                ),
                2,
            )
        else:
            # Generate a regular transaction
            txn_value = round(
                np.random.lognormal(
                    mean=np.log(self.profile["txn_mean"]),
                    sigma=self.profile["txn_sigma"],
                ),
                2,
            )
        return txn_value

    def generate_current_txn(self):
        """
        Generate a single transaction for a customer based on their profile with the current timestamp
        """
        txn_value = self.generate_txn_value()
        txn_timestamp = datetime.now().strftime(self._timestamp_format)
        txn_date = datetime.now().strftime(self._date_format)
        return {
            "txn_id": self._uuid_generator.generate_id(prefix="t_"),
            "txn_timestamp": txn_timestamp,
            "txn_date": txn_date,
            "txn_value": txn_value,
            "txn_fraud": 0,
            "txn_fraud_scenario": 0,
        }

    def generate_batch_txns(
        self, start_date: str = "2024-01-01", num_days: int = 30
    ) -> List[dict]:
        """
        Gererate a list of customer transactions for a given number of days

        Parameters
        -----------
        start_date: str
            The starting date of the transactions in the format 'YYYY-MM-DD'
        num_days: int
            The number of days for which to generate transactions

        Returns
        --------
        List[dict]
            A list of dictionaries where each dictionary represents a transaction
        """
        batch_txns = []
        for day in range(num_days):
            num_txn = np.random.poisson(self.profile["txn_lambda"])
            if num_txn > 0:
                for _ in range(num_txn):
                    # Time of transaction: revolves around noontime, with std 20000 seconds. This is meant to simulate the fact that most transactions should occur during the day (e.g., grocery, gas, other shopping...)
                    time_txn = int(np.random.normal(86400 / 2, 20000))

                    if (time_txn > 0) and (time_txn < 86400):
                        txn_value = self.generate_txn_value()

                        txn_timestamp = datetime.fromtimestamp(
                            time_txn, tz=None
                        ) + timedelta(days=day)

                        txn_timestamp = datetime.strptime(
                            start_date, self._date_format
                        ) + timedelta(seconds=time_txn, days=day)

                        # convert to str and save
                        txn_date = txn_timestamp.strftime(self._date_format)
                        txn_timestamp = txn_timestamp.strftime(self._timestamp_format)
                        batch_txns.append(
                            {
                                "txn_id": self._uuid_generator.generate_id(prefix="t_"),
                                "txn_timestamp": txn_timestamp,
                                "txn_date": txn_date,
                                "txn_value": txn_value,
                                "txn_fraud": 0,
                                "txn_fraud_scenario": 0,
                            }
                        )
        return batch_txns


class CustomerGenerator:
    def __init__(self, group_profiles: dict):
        """
        Initialize the CustomerGenerator with a dictionary of group profiles.

        Parameters:
        -----------
        group_profiles: dict
            A dictionary containing the group profiles for generating customer transactions.
            The dictionary must have the following structure where the key names and their value types are compulsory:
            {
                'name': ['low', 'low-middle', 'middle', 'high-middle', 'high'],
                'txn_mean_low': [5, 20, 40, 60, 80],
                'txn_mean_high': [20, 40, 60, 80, 100],
                'txn_cv_low': [0.3, 0.4, 0.5, 0.6, 0.7],
                'txn_cv_high': [0.4, 0.5, 0.6, 0.7, 0.8],
                'txn_lambda': [0.25, 0.5, 1, 1.5, 2]
            }

        """
        # validate the group_profiles
        try:
            GroupProfiles(**group_profiles)
        except ValidationError as e:
            raise ValueError(f"Invalid group_profiles data: {e}")

        self.group_profiles = self._convert_col_to_row_oriented_profile(
            group_profiles, "name"
        )

    def generate_customer_from_profile(self, profile_name: str):
        """
        Generate a customer object with a specific profile, modelled from the chosen profile name
        """
        assert profile_name in self.group_profiles.keys(), "Profile name not found"
        profile = self.group_profiles[profile_name]
        txn_mean = round(
            np.random.uniform(profile["txn_mean_low"], profile["txn_mean_high"]), 2
        )
        cv = round(np.random.uniform(profile["txn_cv_low"], profile["txn_cv_high"]), 2)
        # txn_sigma = round(self._compute_sigma(txn_mean, cv), 2)
        txn_mu, txn_sigma = compute_lognormal_parameters(txn_mean, cv)
        # round to 2 decimal places
        txn_sigma = txn_sigma
        txn_lambda = profile["txn_lambda"]
        customer_profile = {
            "txn_mean": txn_mean,
            "txn_mu": txn_mu,
            "txn_sigma": txn_sigma,
            "txn_lambda": txn_lambda,
        }
        return Customer(customer_profile)

    def _convert_col_to_row_oriented_profile(
        self, input_dict: dict, key_field: str
    ) -> dict:
        """
        Convert a column-oriented dictionary, which is more concise, to a row-oriented dictionary, which is easier to extract field-specific data from.

        Example:
        --------
        input_dict = {
            'key_field': ['A', 'B', 'C'],
            'field1': [1, 2, 3],
            'field2': [4, 5, 6]
        }
        output_dict = {
            'A': {'field1': 1, 'field2': 4},
            'B': {'field1': 2, 'field2': 5},
            'C': {'field1': 3, 'field2': 6}
        }
        """
        assert (
            key_field in input_dict
        ), f"Key field '{key_field}' not found in input dictionary"
        output_dict = {}
        key_values = input_dict[key_field]
        other_fields = {k: v for k, v in input_dict.items() if k != key_field}

        for i, key in enumerate(key_values):
            output_dict[key] = {
                field: values[i] for field, values in other_fields.items()
            }

        return output_dict


class FraudulentTxnGenerator:
    """
    Class to generate fraudulent transactions for a given customer and date based on a specific scenario.
    """

    def __init__(self):
        self._date_format = "%Y-%m-%d"
        self._timestamp_format = "%Y-%m-%d %H:%M:%S"
        self._uuid_generator = UUIDGenerator(id_length=10)

    def generate_fraudulent_txns(
        self, customer_id: str, scenario: int, date: str
    ) -> List[dict]:
        """
        Generate a batch of fraudulent transactions for a given date.
        Scenario 1: Unusual large transactions scattered through a number of days
        Scenario 2: Large transactions in quick successions with increasing amounts.
        Scenario 3: A small transaction, followed by quick successions of a large amount.

        Parameters:
        -----------
        date: str
            The date of the transactions in the format 'YYYY-MM-DD'. E.g., '2024-01-01'

        Returns:
        --------
        List[dict]
            A list of dictionaries representing the fraudulent transactions
            E.g., [{'txn_id': 't_1', 'txn_timestamp': '2024-01-01 12:00:00', 'txn_value': 100.0, 'txn_fraud': 1, 'txn_fraud_scenario': 1}]
        """

        # Generate a random timestamp within the given date
        start_time = np.random.randint(
            0, 86400
        )  # Random second in the day (0 to 86400)

        # Convert the date string to a datetime object
        date_obj = datetime.strptime(date, self._date_format)

        # Generate the fraudulent transactions
        fraudulent_txns = []
        txn_value_increment = random.choice(range(500, 2000, 500))

        if scenario == 1:
            compromised_days = np.random.randint(
                5, 14
            )  # Can be converted to user's input later
            current_date = date_obj
            # for each day within the compromised days, generate a few transactions with large values
            # the large values are arbitrarily chosen to be "nice" numbers
            for _ in range(compromised_days):
                num_txns_day = np.random.randint(1, 3)
                for i in range(num_txns_day):
                    # txn_timestamp is a random time within the day
                    txn_timestamp = current_date + timedelta(
                        seconds=np.random.randint(0, 86400)
                    )
                    txn_date = txn_timestamp.strftime(self._date_format)
                    txn_timestamp = txn_timestamp.strftime(self._timestamp_format)
                    txn_value = round(
                        random.choice(range(500, 2000, 500)), 2
                    )  # Example transaction value

                    fraudulent_txns.append(
                        {
                            "customer_id": customer_id,
                            "txn_id": self._uuid_generator.generate_id(prefix="t_"),
                            "txn_timestamp": txn_timestamp,
                            "txn_date": txn_date,
                            "txn_value": txn_value,
                            "txn_fraud": 1,
                            "txn_fraud_scenario": 1,
                        }
                    )
                current_date += timedelta(days=1)

        elif scenario == 2:
            num_txns = np.random.randint(5, 10)
            interval_seconds = np.random.randint(1, 5) * 60
            for i in range(num_txns):
                txn_timestamp = date_obj + timedelta(
                    seconds=start_time + i * np.random.randint(1, interval_seconds)
                )
                txn_date = txn_timestamp.strftime(self._date_format)
                txn_timestamp = txn_timestamp.strftime(self._timestamp_format)
                txn_value = round(
                    (i + 1) * txn_value_increment, 2
                )  # Example transaction value
                fraudulent_txns.append(
                    {
                        "customer_id": customer_id,
                        "txn_id": self._uuid_generator.generate_id(prefix="t_"),
                        "txn_timestamp": txn_timestamp,
                        "txn_date": txn_date,
                        "txn_value": txn_value,
                        "txn_fraud": 1,
                        "txn_fraud_scenario": 2,
                    }
                )
        elif scenario == 3:
            num_txns = np.random.randint(5, 10)
            interval_seconds = np.random.randint(1, 5) * 60
            for i in range(num_txns):
                txn_timestamp = date_obj + timedelta(
                    seconds=start_time + i * np.random.randint(1, interval_seconds)
                )
                txn_date = txn_timestamp.strftime(self._date_format)
                txn_timestamp = txn_timestamp.strftime(self._timestamp_format)
                if i == 0:
                    txn_value = round(random.uniform(5, 10), 2)
                else:
                    txn_value = round(
                        txn_value_increment, 2
                    )  # use a fixed rounded number (for clearer difference from scenario 2)
                fraudulent_txns.append(
                    {
                        "customer_id": customer_id,
                        "txn_id": self._uuid_generator.generate_id(prefix="t_"),
                        "txn_timestamp": txn_timestamp,
                        "txn_date": txn_date,
                        "txn_value": txn_value,
                        "txn_fraud": 1,
                        "txn_fraud_scenario": 3,
                    }
                )
        else:
            raise ValueError("Invalid scenario number. Choose 1, 2 or 3.")
        return fraudulent_txns
