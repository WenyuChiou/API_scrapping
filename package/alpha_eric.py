#%%
import numpy as np
import pandas as pd

#%%
class AlphaFactory:
    def __init__(self, data):
        """
        Initialize with futures data including open, high, low, close, volume, spread.
        :param data: pd.DataFrame with columns ['open', 'High', 'low', 'close', 'Volume', 'Spread']
        """
        self.data = data

    def alpha01(self, days=list, par=None, type=None):
        """
        The alpha is derived from expected utility theory.
        To assess the utility of risk-averse or risk-seeking people when facing gains and loss,
        We apply the utility function using the percentage of difference of close price at t = n + interval and t = n.
        
        Two scenarios are being considered:
        
        If X is above and below 0, the utility function will be u(x) = log(x) or x**(1/2)
        If X is below 0, the utility function will be u(x) = -x**2

        Parameter:
        - days (list): the time intervals for calculating differences
        - par (dict, optional): the parameter of utility function
            example: {'alpha':0.88, 'beta': 0.88, 'theta': 2.25}
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        alpha = pd.DataFrame()
        
        if par:
            A = par['alpha']
            B = par['beta']
            theta = par['theta']
        else:
            A, B, theta = 1/2, 2, 1
            
        if type is None:
            
            for day in days:
                X = self.data['close'].rolling(window=day).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
                self.data[f'utility_{day}_type1'] = X.apply(lambda x: x**(A) if x >= 0 else -theta * (x ** (B) ))
        else:
            for day in days:
                X = self.data['close'].rolling(window=day).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
                self.data[f'utility_{day}_type2'] = X.apply(lambda x: np.log(x) if x >= 0 else -theta * (x ** (B) ))
                
        return self.data

    def alpha02(self, days=list, weight=0.5):
        """
        The alpha is derived from prospect theory where people value gains and losses differently.
        We assess the reference dependence effect where people overweight losses compared to gains.
        Here, we apply a weighted moving average of gains and losses in price movements.
        
        Parameter:
        - days (list): the time intervals for calculating differences
        - weight (float): weight assigned to losses vs gains (default is 0.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_diff = self.data['close'].diff(day)
            self.data[f'prospect_{day}'] = price_diff.apply(lambda x: weight * x if x < 0 else (1 - weight) * x)
        return self.data

    def alpha03(self, days=list, risk_aversion=1.5):
        """
        The alpha is derived from cumulative prospect theory to capture risk aversion behavior.
        
        The utility is calculated using a power function where the parameter represents risk aversion.
        Higher risk aversion values will weigh negative returns more heavily.
        
        Parameter:
        - days (list): the time intervals for calculating differences
        - risk_aversion (float): risk aversion parameter (default is 1.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_diff = self.data['close'].diff(day)
            self.data[f'risk_aversion_{day}'] = price_diff.apply(lambda x: (abs(x) ** risk_aversion) * (-1 if x < 0 else 1))
        return self.data

    def alpha04(self, days=list):
        """
        The alpha is derived from the concept of mental accounting, where traders treat each day's profit or loss separately.
        Here, we calculate the running difference and evaluate each separately.
        
        Parameter:
        - days (list): the time intervals for calculating differences
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            self.data[f'mental_accounting_{day}'] = self.data['close'].diff(day)
        return self.data

    def alpha05(self, days=list):
        """
        The alpha is derived from loss aversion theory, where losses are weighted more heavily than gains.
        The focus is on identifying large drops in prices to signal strong emotional reactions.
        
        Parameter:
        - days (list): the time intervals for calculating differences
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            self.data[f'loss_aversion_{day}'] = self.data['close'].diff(day).apply(lambda x: x if x < 0 else 0)
        return self.data

    def alpha06(self, days=list, threshold=1.5):
        """
        The alpha is based on overconfidence theory where traders overreact to positive changes in price.
        We capture sudden price increases beyond a threshold to indicate overconfidence.
        
        Parameter:
        - days (list): the time intervals for calculating differences
        - threshold (float): threshold for price change to signal overconfidence (default is 1.5%)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_change = self.data['close'].pct_change(day) * 100
            self.data[f'overconfidence_{day}'] = price_change.apply(lambda x: x if x > threshold else 0)
        return self.data

    def alpha07(self, days=list, regret_aversion=2):
        """
        The alpha is derived from regret aversion theory where traders avoid realizing losses due to regret.
        We track the underperformance relative to previous highs to indicate potential regret aversion.
        
        Parameter:
        - days (list): the time intervals for calculating previous highs
        - regret_aversion (float): scaling factor for underperformance (default is 2)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            rolling_high = self.data['close'].rolling(window=day).max()
            self.data[f'regret_aversion_{day}'] = (rolling_high - self.data['close']) * regret_aversion
        return self.data

    def alpha08(self, days=list, euphoria_threshold=2):
        """
        The alpha is based on the theory of euphoria, where traders exhibit irrational exuberance during price rallies.
        We assess the magnitude of consecutive price increases exceeding a threshold to indicate euphoria.
        
        Parameter:
        - days (list): the time intervals for calculating price changes
        - euphoria_threshold (float): threshold for consecutive gains to signal euphoria (default is 2%)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_change = self.data['close'].pct_change(day) * 100
            self.data[f'euphoria_{day}'] = price_change.apply(lambda x: x if x > euphoria_threshold else 0)
        return self.data

    def alpha09(self, days=list):
        """
        The alpha is derived from anchoring bias, where traders are influenced by recent price levels as reference points.
        We calculate the deviation of the current price from the moving average to capture anchoring effects.
        
        Parameter:
        - days (list): the time intervals for calculating moving average
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            moving_avg = self.data['close'].rolling(window=day).mean()
            self.data[f'anchoring_{day}'] = self.data['close'] - moving_avg
        return self.data

    def alpha10(self, days=list, disposition_effect=1.2):
        """
        The alpha is derived from disposition effect, where traders are more inclined to sell assets that have increased in value.
        We measure the magnitude of gains relative to the moving average as a signal for potential selling pressure.
        
        Parameter:
        - days (list): the time intervals for calculating differences
        - disposition_effect (float): scaling factor for gains (default is 1.2)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_diff = self.data['close'].diff(day)
            self.data[f'disposition_effect_{day}'] = price_diff.apply(lambda x: disposition_effect * x if x > 0 else 0)
        return self.data

    def alpha11(self, days=list, uncertainty_factor=1.5):
        """
        The alpha is based on ambiguity aversion, where traders avoid uncertain outcomes.
        We capture the impact of increased price volatility on trader behavior.
        
        Parameter:
        - days (list): the time intervals for calculating rolling standard deviation
        - uncertainty_factor (float): factor to scale the uncertainty impact (default is 1.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            rolling_std = self.data['close'].rolling(window=day).std()
            self.data[f'ambiguity_aversion_{day}'] = rolling_std * uncertainty_factor
        return self.data

    def alpha12(self, days=list, adjustment_factor=0.1):
        """
        The alpha is derived from adaptive expectations theory, where traders adjust their expectations based on recent trends.
        We apply an exponential smoothing to recent prices to model adaptive behavior.
        
        Parameter:
        - days (list): the time intervals for calculating exponential moving average
        - adjustment_factor (float): smoothing factor for expectations (default is 0.1)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            self.data[f'adaptive_expectations_{day}'] = self.data['close'].ewm(span=day, adjust=False).mean() * adjustment_factor
        return self.data

    def alpha13(self, days=list, noise_factor=0.02):
        """
        The alpha is derived from the concept of bounded rationality, where traders make decisions with limited information.
        We add random noise to the price movements to simulate bounded rationality behavior.
        
        Parameter:
        - days (list): the time intervals for calculating price changes
        - noise_factor (float): factor to scale the random noise (default is 0.02)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            random_noise = np.random.normal(0, noise_factor, len(self.data))
            self.data[f'bounded_rationality_{day}'] = self.data['close'].pct_change(day) + random_noise
        return self.data

    def alpha14(self, days=list, optimism_bias=1.1):
        """
        The alpha is based on optimism bias, where traders overestimate positive outcomes.
        We apply a scaling factor to positive price changes to capture optimistic behavior.
        
        Parameter:
        - days (list): the time intervals for calculating price changes
        - optimism_bias (float): factor to scale positive price changes (default is 1.1)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_change = self.data['close'].pct_change(day)
            self.data[f'optimism_bias_{day}'] = price_change.apply(lambda x: x * optimism_bias if x > 0 else x)
        return self.data

    def alpha15(self, days=list, volatility_scaler=0.05):
        """
        The alpha is derived from the concept of risk perception, where traders adjust their behavior based on perceived risk.
        We use a volatility-adjusted price change to model risk perception.
        
        Parameter:
        - days (list): the time intervals for calculating volatility
        - volatility_scaler (float): factor to scale the volatility impact (default is 0.05)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            rolling_volatility = self.data['close'].rolling(window=day).std()
            self.data[f'risk_perception_{day}'] = self.data['close'].pct_change(day) * (1 + volatility_scaler * rolling_volatility)
        return self.data

    def alpha13(self, days=list, noise_factor=0.02):
        """
        The alpha is derived from the concept of bounded rationality, where traders make decisions with limited information.
        We add random noise to the price movements to simulate bounded rationality behavior.
        
        Parameter:
        - days (list): the time intervals for calculating price changes
        - noise_factor (float): factor to scale the random noise (default is 0.02)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            random_noise = np.random.normal(0, noise_factor, len(self.data))
            self.data[f'bounded_rationality_{day}'] = self.data['close'].pct_change(day) + random_noise
        return self.data

    def alpha14(self, days=list, optimism_bias=1.1):
        """
        The alpha is based on optimism bias, where traders overestimate positive outcomes.
        We apply a scaling factor to positive price changes to capture optimistic behavior.
        
        Parameter:
        - days (list): the time intervals for calculating price changes
        - optimism_bias (float): factor to scale positive price changes (default is 1.1)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_change = self.data['close'].pct_change(day)
            self.data[f'optimism_bias_{day}'] = price_change.apply(lambda x: x * optimism_bias if x > 0 else x)
        return self.data

    def alpha15(self, days=list, volatility_scaler=0.05):
        """
        The alpha is derived from the concept of risk perception, where traders adjust their behavior based on perceived risk.
        We use a volatility-adjusted price change to model risk perception.
        
        Parameter:
        - days (list): the time intervals for calculating volatility
        - volatility_scaler (float): factor to scale the volatility impact (default is 0.05)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            rolling_volatility = self.data['close'].rolling(window=day).std()
            self.data[f'risk_perception_{day}'] = self.data['close'].pct_change(day) * (1 + volatility_scaler * rolling_volatility)
        return self.data

    def alpha16(self, days=list, conformity_factor=0.5):
        """
        The alpha is based on herd behavior, where traders follow the actions of the majority.
        We use the average price movement over a period to model conformity.
        
        Parameter:
        - days (list): the time intervals for calculating average price movement
        - conformity_factor (float): factor to scale conformity behavior (default is 0.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            avg_movement = self.data['close'].rolling(window=day).mean()
            self.data[f'herd_behavior_{day}'] = (self.data['close'] - avg_movement) * conformity_factor
        return self.data

    def alpha17(self, days=list, emotional_intensity=1.5):
        """
        The alpha is based on emotional intensity theory, where traders' decisions are influenced by the intensity of price changes.
        We assess the rate of change of price with an intensity multiplier.
        
        Parameter:
        - days (list): the time intervals for calculating rate of change
        - emotional_intensity (float): factor to scale the intensity of the reaction (default is 1.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        rate_of_change_list = []
        for day in days:
            rate_of_change = self.data['close'].pct_change(day)
            rate_of_change_list.append(rate_of_change * emotional_intensity)
        alpha_data = pd.concat(rate_of_change_list, axis=1)
        alpha_data.columns = [f'emotional_intensity_{day}' for day in days]
        self.data = pd.concat([self.data, alpha_data], axis=1)
        return self.data

    def alpha18(self, days=list, regret_bias=1.3):
        """
        The alpha is derived from regret theory, where traders experience regret after making suboptimal decisions.
        We model this by measuring deviations from the highest price within the period.
        
        Parameter:
        - days (list): the time intervals for calculating maximum price
        - regret_bias (float): factor to scale the regret impact (default is 1.3)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            rolling_max = self.data['close'].rolling(window=day).max()
            self.data[f'regret_bias_{day}'] = (rolling_max - self.data['close']) * regret_bias
        return self.data

    def alpha19(self, days=list, gambler_fallacy_factor=0.2):
        """
        The alpha is based on the gambler's fallacy, where traders believe that future prices will reverse after a streak.
        We assess the impact of consecutive gains or losses on traders' expectations.
        
        Parameter:
        - days (list): the time intervals for calculating streaks
        - gambler_fallacy_factor (float): factor to scale the gambler's fallacy effect (default is 0.2)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            streak = self.data['close'].diff().rolling(window=day).apply(lambda x: sum(x > 0) - sum(x < 0))
            self.data[f'gambler_fallacy_{day}'] = streak * gambler_fallacy_factor
        return self.data

    def alpha20(self, days=list, fear_index_factor=0.4):
        """
        The alpha is derived from fear index concepts, where traders react strongly to significant downward movements.
        We use a factor to enhance the effect of negative returns during high volatility periods.
        
        Parameter:
        - days (list): the time intervals for calculating negative returns
        - fear_index_factor (float): factor to scale the fear response (default is 0.4)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            negative_return = self.data['close'].pct_change(day).apply(lambda x: x if x < 0 else 0)
            rolling_volatility = self.data['close'].rolling(window=day).std()
            self.data[f'fear_index_{day}'] = negative_return * (1 + fear_index_factor * rolling_volatility)
        return self.data

    def alpha21(self, days=list, overconfidence_factor=0.3):
        """
        The alpha is based on overconfidence bias, where traders overestimate their ability to predict future price movements.
        We calculate the deviation of closing price from the high and apply a scaling factor for overconfidence.
        
        Parameter:
        - days (list): the time intervals for calculating the high deviation
        - overconfidence_factor (float): factor to scale the overconfidence impact (default is 0.3)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            deviation = self.data['high'] - self.data['close']
            self.data[f'overconfidence_bias_{day}'] = deviation.rolling(window=day).mean() * overconfidence_factor
        return self.data

    def alpha22(self, days=list, recency_bias=0.5):
        """
        The alpha is derived from recency bias, where traders give more weight to recent price movements.
        We use exponentially weighted moving average on both close and volume to model recency effects.
        
        Parameter:
        - days (list): the time intervals for calculating the effect
        - recency_bias (float): bias factor to scale recent effects (default is 0.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            close_ewma = self.data['close'].ewm(span=day, adjust=False).mean()
            volume_ewma = self.data['volume'].ewm(span=day, adjust=False).mean()
            self.data[f'recency_bias_{day}'] = (close_ewma * volume_ewma) * recency_bias
        return self.data

    def alpha23(self, days=list, loss_aversion_factor=2):
        """
        The alpha is based on loss aversion, where losses have a larger impact on traders than gains.
        We enhance the negative price changes by a loss aversion factor.
        
        Parameter:
        - days (list): the time intervals for calculating price changes
        - loss_aversion_factor (float): factor to amplify losses (default is 2)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_change = self.data['close'].pct_change(day)
            self.data[f'loss_aversion_{day}'] = price_change.apply(lambda x: x * loss_aversion_factor if x < 0 else x)
        return self.data

    def alpha24(self, days=list, cognitive_dissonance_factor=0.7):
        """
        The alpha is based on cognitive dissonance, where traders justify bad decisions by focusing on non-price information.
        We calculate the deviation between open and close price with a scaling factor.
        
        Parameter:
        - days (list): the time intervals for calculating open-close deviation
        - cognitive_dissonance_factor (float): factor to scale dissonance behavior (default is 0.7)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            open_close_deviation = (self.data['close'] - self.data['open']).rolling(window=day).mean()
            self.data[f'cognitive_dissonance_{day}'] = open_close_deviation * cognitive_dissonance_factor
        return self.data

    def alpha25(self, days=list, random_shock_factor=0.1):
        """
        The alpha is inspired by the theory of market surprises, where sudden news shocks affect trader behavior.
        We introduce a random shock element to the volume to simulate such events.
        
        Parameter:
        - days (list): the time intervals for calculating shocks
        - random_shock_factor (float): scaling factor for shock impact (default is 0.1)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            random_shock = np.random.uniform(-1, 1, len(self.data)) * random_shock_factor
            self.data[f'random_shock_{day}'] = self.data['volume'].rolling(window=day).mean() * (1 + random_shock)
        return self.data

    def alpha26(self, days=list, endowment_effect=0.8):
        """
        The alpha is derived from the endowment effect, where traders overvalue owned assets compared to non-owned.
        We take the deviation of closing price from its mean and apply an endowment factor.
        
        Parameter:
        - days (list): the time intervals for calculating deviation from mean
        - endowment_effect (float): scaling factor for endowment (default is 0.8)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            mean_price = self.data['close'].rolling(window=day).mean()
            deviation = self.data['close'] - mean_price
            self.data[f'endowment_effect_{day}'] = deviation * endowment_effect
        return self.data

    def alpha27(self, days=list, anchoring_factor=0.6):
        """
        The alpha is based on anchoring, where traders rely too heavily on the initial piece of information.
        We use the open price as an anchor and measure subsequent price changes against it.
        
        Parameter:
        - days (list): the time intervals for calculating anchoring impact
        - anchoring_factor (float): scaling factor for anchoring (default is 0.6)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            self.data[f'anchoring_effect_{day}'] = (self.data['close'] - self.data['open'].shift(day)) * anchoring_factor
        return self.data

    def alpha28(self, days=list, expectation_discrepancy=1.2):
        """
        The alpha is based on expectation discrepancy, where actual price movements differ from anticipated trends.
        We use both the high and low prices to assess discrepancy from expectations.
        
        Parameter:
        - days (list): the time intervals for calculating discrepancy
        - expectation_discrepancy (float): scaling factor for expectation discrepancy (default is 1.2)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            avg_high_low = (self.data['high'] + self.data['low']) / 2
            expected_movement = avg_high_low.rolling(window=day).mean()
            self.data[f'expectation_discrepancy_{day}'] = (self.data['close'] - expected_movement) * expectation_discrepancy
        return self.data

    def alpha29(self, days=list, fear_of_missing_out=0.9):
        """
        The alpha is based on FOMO, where traders react irrationally to fear of missing out on profit opportunities.
        We measure how closing price deviates from the recent high and scale by the FOMO factor.
        
        Parameter:
        - days (list): the time intervals for calculating recent high
        - fear_of_missing_out (float): scaling factor for FOMO (default is 0.9)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            recent_high = self.data['high'].rolling(window=day).max()
            self.data[f'fomo_{day}'] = (recent_high - self.data['close']) * fear_of_missing_out
        return self.data

    def alpha30(self, days=list, crowd_fear_index=0.7):
        """
        The alpha is inspired by crowd fear, where traders amplify their behavior during volatile times.
        We use both volume and spread to assess the level of crowd fear during high volatility.
        
        Parameter:
        - days (list): the time intervals for calculating crowd fear
        - crowd_fear_index (float): scaling factor for crowd fear impact (default is 0.7)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            spread_volatility = self.data['spread'].rolling(window=day).std()
            volume_average = self.data['volume'].rolling(window=day).mean()
            self.data[f'crowd_fear_{day}'] = (spread_volatility * volume_average) * crowd_fear_index
        return self.data

    def alpha31(self, days=list, market_sentiment_factor=0.5):
        """
        The alpha is based on market sentiment where traders react differently to upward vs downward trends.
        We apply a factor to the difference between open and close, adjusted for trend direction.
        
        Parameter:
        - days (list): the time intervals for calculating sentiment effect
        - market_sentiment_factor (float): factor to scale market sentiment impact (default is 0.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            trend_direction = np.sign(self.data['close'] - self.data['open'])
            self.data[f'market_sentiment_{day}'] = (self.data['close'] - self.data['open']).rolling(window=day).mean() * trend_direction * market_sentiment_factor
        return self.data

    def alpha32(self, days=list, randomness_factor=0.03):
        """
        The alpha is derived from the noise trader hypothesis, where irrational traders create market noise.
        We add random Gaussian noise to price changes to simulate noise trading behavior.
        
        Parameter:
        - days (list): the time intervals for calculating price change
        - randomness_factor (float): standard deviation of the Gaussian noise added (default is 0.03)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            random_noise = np.random.normal(0, randomness_factor, len(self.data))
            self.data[f'noise_trader_{day}'] = self.data['close'].pct_change(day) + random_noise
        return self.data

    def alpha33(self, days=list, asymmetric_volatility_factor=0.6):
        """
        The alpha is derived from the leverage effect, where volatility is higher during downward trends.
        We apply an asymmetric factor to volatility depending on the direction of price change.

        Parameter:
        - days (list): the time intervals for calculating volatility
        - asymmetric_volatility_factor (float): factor to scale volatility asymmetrically (default is 0.6)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        alpha_data = []
        for day in days:
            price_change = self.data['close'].pct_change(day)
            rolling_volatility = self.data['close'].rolling(window=day).std()
            alpha_data.append(rolling_volatility * (1 + asymmetric_volatility_factor * (price_change < 0)))
        alpha_df = pd.concat(alpha_data, axis=1)
        alpha_df.columns = [f'asymmetric_volatility_{day}' for day in days]
        self.data = pd.concat([self.data, alpha_df], axis=1)
        return self.data


    def alpha34(self, days=list, anchoring_bias_factor=0.4):
        """
        The alpha is based on anchoring bias, where traders rely on recent reference points.
        We use the previous day's close as an anchor and measure deviations from it.
        
        Parameter:
        - days (list): the time intervals for calculating anchor deviations
        - anchoring_bias_factor (float): factor to scale the anchoring effect (default is 0.4)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            anchor = self.data['close'].shift(day)
            self.data[f'anchoring_bias_{day}'] = (self.data['close'] - anchor) * anchoring_bias_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha35(self, days=list, self_attribution_factor=0.25):
        """
        The alpha is based on self-attribution bias, where traders attribute success to their own skill.
        We calculate gains from low to high prices and apply a factor to model overconfidence in gains.
        
        Parameter:
        - days (list): the time intervals for calculating gains
        - self_attribution_factor (float): factor to scale self-attribution impact (default is 0.25)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            gains = (self.data['high'] - self.data['low']).rolling(window=day).sum()
            self.data[f'self_attribution_{day}'] = gains * self_attribution_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha36(self, days=list, regret_aversion_factor=0.8):
        """
        The alpha is based on regret aversion, where traders avoid realizing losses to prevent regret.
        We measure the drawdown from recent highs and apply a factor to amplify this behavior.
        
        Parameter:
        - days (list): the time intervals for calculating drawdown
        - regret_aversion_factor (float): factor to scale regret aversion impact (default is 0.8)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            recent_high = self.data['high'].rolling(window=day).max()
            drawdown = recent_high - self.data['close']
            self.data[f'regret_aversion_{day}'] = drawdown * regret_aversion_factor
        return self.data

    def alpha37(self, days=list, disposition_effect_factor=0.9):
        """
        The alpha is based on the disposition effect, where traders are more likely to sell assets that have gained value.
        We calculate the return from open to close and apply a factor to model this effect.

        Parameter:
        - days (list): the time intervals for calculating returns
        - disposition_effect_factor (float): factor to scale disposition effect impact (default is 0.9)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            return_open_close = (self.data['close'] - self.data['open']).rolling(window=day).mean()
            new_columns[f'disposition_effect_2_{day}'] = return_open_close * disposition_effect_factor
        
        # Use pd.concat to add new columns to the DataFrame
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data


    def alpha38(self, days=list, mental_accounting_factor=0.3):
        """
        The alpha is based on mental accounting, where traders separate their investments into different 'accounts'.
        We use the difference between volume and spread to model traders' behavior of categorizing trades.

        Parameter:
        - days (list): the time intervals for calculating volume-spread difference
        - mental_accounting_factor (float): factor to scale mental accounting impact (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            volume_spread_diff = (self.data['volume'] - self.data['spread']).rolling(window=day).mean()
            new_columns[f'mental_accounting_{day}'] = volume_spread_diff * mental_accounting_factor

        # Use pd.concat to add new columns to the DataFrame
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data


    def alpha39(self, days=list, endowment_effect_factor=1.1):
        """
        The alpha is derived from the endowment effect, where traders value assets they own more highly.
        We measure the deviation of the close price from the average of open, high, and low prices.
        
        Parameter:
        - days (list): the time intervals for calculating the deviation
        - endowment_effect_factor (float): factor to scale the endowment effect impact (default is 1.1)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            avg_price = (self.data['open'] + self.data['high'] + self.data['low']) / 3
            self.data[f'endowment_effect_{day}'] = (self.data['close'] - avg_price).rolling(window=day).mean() * endowment_effect_factor
        return self.data

    def alpha40(self, days=list, random_walk_factor=0.05):
        """
        The alpha is derived from the random walk theory, where price changes are assumed to be random.
        We add a stochastic term to the price movements to simulate the randomness.

        Parameter:
        - days (list): the time intervals for calculating random walk effect
        - random_walk_factor (float): factor to scale the randomness (default is 0.05)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            stochastic_term = np.random.normal(0, random_walk_factor, len(self.data))
            new_columns[f'random_walk_{day}'] = self.data['close'].pct_change(day) + stochastic_term

        # Use pd.concat to add new columns to the DataFrame
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha41(self, days=list, anchoring_factor=1.5):
        """
        The alpha is derived from anchoring bias, where traders rely too heavily on the first piece of information (price).
        We take the average of open prices and apply an adjustment to model the anchoring effect.
        
        Parameter:
        - days (list): time intervals for calculating anchoring
        - anchoring_factor (float): factor to scale the effect of anchoring (default is 1.5)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            anchoring_effect = self.data['open'].rolling(window=day).mean() * anchoring_factor
            new_columns[f'anchoring_ori_{day}'] = anchoring_effect
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha42(self, days=list, mean_reversion_factor=0.8):
        """
        The alpha is based on mean reversion theory, which assumes that prices will revert to their historical average.
        We calculate the deviation from the mean close price over time.
        
        Parameter:
        - days (list): time intervals for calculating mean reversion
        - mean_reversion_factor (float): factor to scale mean reversion impact (default is 0.8)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            mean_price = self.data['close'].rolling(window=day).mean()
            deviation = (self.data['close'] - mean_price) * mean_reversion_factor
            new_columns[f'mean_reversion_{day}'] = deviation
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha43(self, days=list, regret_aversion_factor=1.1):
        """
        The alpha is based on regret aversion theory, where traders avoid selling assets to prevent regret.
        We calculate the maximum of high prices over a period, adjusted by a regret factor.
        
        Parameter:
        - days (list): time intervals for calculating regret
        - regret_aversion_factor (float): factor to scale regret aversion (default is 1.1)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            max_high = self.data['high'].rolling(window=day).max()
            new_columns[f'regret_aversion_{day}'] = max_high * regret_aversion_factor
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha44(self, days=list, stochastic_factor=0.02):
        """
        The alpha incorporates a stochastic element to simulate unpredictable market behavior.
        We add a random normal term to the rate of change in close prices.
        
        Parameter:
        - days (list): time intervals for calculating stochastic effect
        - stochastic_factor (float): factor for randomness (default is 0.02)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            stochastic_term = np.random.normal(0, stochastic_factor, len(self.data))
            new_columns[f'stochastic_alpha_{day}'] = self.data['close'].pct_change(day) + stochastic_term
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha45(self, days=list, sunspot_factor=0.01):
        """
        The alpha is based on the "sunspot theory," suggesting that unrelated events like sunspots can affect market sentiment.
        We add a trigonometric function to introduce an oscillatory component to the price changes.
        
        Parameter:
        - days (list): time intervals for calculating sunspot effect
        - sunspot_factor (float): factor for the sunspot effect (default is 0.01)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            oscillatory_term = np.sin(np.linspace(0, 2 * np.pi * day, len(self.data))) * sunspot_factor
            new_columns[f'sunspot_alpha_{day}'] = self.data['close'].pct_change(day) + oscillatory_term
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha46(self, days=list, confidence_index=0.05):
        """
        The alpha is based on trader confidence, using volume as a proxy for confidence.
        We multiply volume with a change in spread over time.
        
        Parameter:
        - days (list): time intervals for calculating confidence index
        - confidence_index (float): factor to scale the confidence impact (default is 0.05)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            volume_change = self.data['volume'].pct_change(day)
            spread_change = self.data['spread'].rolling(window=day).mean()
            new_columns[f'confidence_index_{day}'] = volume_change * spread_change * confidence_index
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha47(self, days=list, herding_factor=0.3):
        """
        The alpha is derived from herding behavior, where traders follow the majority.
        We calculate the correlation between volume and closing price to determine herding effects.
        
        Parameter:
        - days (list): time intervals for calculating herding behavior
        - herding_factor (float): factor to scale the herding impact (default is 0.3)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            volume_close_corr = self.data['volume'].rolling(window=day).corr(self.data['close']) * herding_factor
            new_columns[f'herding_{day}'] = volume_close_corr
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha48(self, days=list, expectation_bias_factor=0.25):
        """
        The alpha models expectation bias, where traders set expectations based on past prices.
        We calculate the expected high price based on past averages and compare it to current high.
        
        Parameter:
        - days (list): time intervals for calculating expectation bias
        - expectation_bias_factor (float): factor to scale the bias impact (default is 0.25)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            expected_high = self.data['high'].rolling(window=day).mean()
            new_columns[f'expectation_bias_{day}'] = (self.data['high'] - expected_high) * expectation_bias_factor
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha49(self, days=list, asymmetric_info_factor=0.6):
        """
        The alpha is based on the theory of asymmetric information, where some traders have more information.
        We calculate the difference between high and low prices, multiplied by volume to represent the information effect.
        
        Parameter:
        - days (list): time intervals for calculating asymmetric information effect
        - asymmetric_info_factor (float): factor to scale the effect (default is 0.6)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            high_low_diff = (self.data['high'] - self.data['low']).rolling(window=day).mean()
            volume_effect = self.data['volume'].rolling(window=day).mean()
            new_columns[f'asymmetric_info_{day}'] = high_low_diff * volume_effect * asymmetric_info_factor
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data

    def alpha50(self, days=list, random_confidence_factor=0.4):
        """
        The alpha introduces randomness to simulate the uncertain nature of market confidence.
        We add a uniformly distributed random term to the percentage change in closing price.
        
        Parameter:
        - days (list): time intervals for calculating random confidence effect
        - random_confidence_factor (float): factor to scale the random confidence (default is 0.4)
        
        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        new_columns = {}
        for day in days:
            random_term = np.random.uniform(-random_confidence_factor, random_confidence_factor, len(self.data))
            new_columns[f'random_confidence_{day}'] = self.data['close'].pct_change(day) + random_term
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        self.data = self.data.copy()
        return self.data
    
    def alpha51(self, days=list, emotional_intensity_factor=0.7):
        """
        The alpha is based on emotional intensity linked to trading volume, where high volume represents higher market emotion.
        We assess the intensity of volume relative to price change.

        Parameter:
        - days (list): the time intervals for calculating volume impact
        - emotional_intensity_factor (float): factor to scale the intensity of the reaction (default is 0.7)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            volume_change = self.data['volume'].pct_change(day)
            price_change = self.data['close'].pct_change(day)
            self.data[f'emotional_intensity_{day}'] = (volume_change * price_change) * emotional_intensity_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha52(self, days=list, panic_volume_threshold=1.5):
        """
        The alpha is based on panic selling theory, where unusually high volume suggests panic.
        We measure instances where volume exceeds a multiple of its moving average.

        Parameter:
        - days (list): the time intervals for calculating volume averages
        - panic_volume_threshold (float): threshold multiplier for identifying panic (default is 1.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            rolling_avg_volume = self.data['volume'].rolling(window=day).mean()
            self.data[f'panic_volume_{day}'] = np.where(self.data['volume'] > panic_volume_threshold * rolling_avg_volume, 1, 0)
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha53(self, days=list, fear_of_missing_out_factor=0.5):
        """
        The alpha is based on the Fear of Missing Out (FOMO), where traders react strongly to rapid price increases.
        We use the interaction of volume and upward price movement to quantify FOMO.

        Parameter:
        - days (list): the time intervals for calculating FOMO effect
        - fear_of_missing_out_factor (float): factor to scale FOMO impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            upward_movement = self.data['close'].pct_change(day).apply(lambda x: x if x > 0 else 0)
            fomo_intensity = (upward_movement * self.data['volume']) * fear_of_missing_out_factor
            self.data[f'fomo_{day}'] = fomo_intensity
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha54(self, days=list, herd_behavior_factor=0.8):
        """
        The alpha is based on herd behavior theory, where traders follow the crowd during high volume periods.
        We assess herd behavior by calculating the price impact during high volume events.

        Parameter:
        - days (list): the time intervals for calculating herd behavior
        - herd_behavior_factor (float): factor to scale herd behavior impact (default is 0.8)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            volume_zscore = (self.data['volume'] - self.data['volume'].rolling(window=day).mean()) / self.data['volume'].rolling(window=day).std()
            price_change = self.data['close'].pct_change(day)
            self.data[f'herd_behavior_{day}'] = (volume_zscore * price_change) * herd_behavior_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha55(self, days=list, regret_avoidance_factor=0.4):
        """
        The alpha is based on regret avoidance, where traders avoid actions that could lead to regret.
        We model this by evaluating volume during periods of negative price movement.

        Parameter:
        - days (list): the time intervals for calculating regret avoidance impact
        - regret_avoidance_factor (float): factor to scale regret avoidance impact (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            negative_price_change = self.data['close'].pct_change(day).apply(lambda x: x if x < 0 else 0)
            regret_intensity = (negative_price_change * self.data['volume']) * regret_avoidance_factor
            self.data[f'regret_avoidance_{day}'] = regret_intensity
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha56(self, days=list, overconfidence_factor=0.6):
        """
        The alpha is based on overconfidence bias, where traders overestimate their ability during times of high returns.
        We use volume to measure the market's confidence when returns are positive.

        Parameter:
        - days (list): the time intervals for calculating overconfidence effect
        - overconfidence_factor (float): factor to scale the overconfidence effect (default is 0.6)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            positive_return = self.data['close'].pct_change(day).apply(lambda x: x if x > 0 else 0)
            overconfidence = (positive_return * self.data['volume']) * overconfidence_factor
            self.data[f'overconfidence_{day}'] = overconfidence
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha57(self, days=list, optimism_pessimism_factor=0.3):
        """
        The alpha is based on the balance between optimism and pessimism, where high volume during price increase represents optimism and vice versa.
        We assess the relationship between volume and price direction.

        Parameter:
        - days (list): the time intervals for calculating optimism or pessimism impact
        - optimism_pessimism_factor (float): factor to scale optimism or pessimism (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_change = self.data['close'].pct_change(day)
            volume_impact = self.data['volume'].rolling(window=day).mean()
            self.data[f'optimism_pessimism_{day}'] = (price_change * volume_impact) * optimism_pessimism_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha58(self, days=list, loss_aversion_factor=0.5):
        """
        The alpha is based on loss aversion, where traders react more strongly to losses than gains.
        We assess the interaction of volume during downward price movement to quantify loss aversion.

        Parameter:
        - days (list): the time intervals for calculating loss aversion effect
        - loss_aversion_factor (float): factor to scale loss aversion impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            downward_movement = self.data['close'].pct_change(day).apply(lambda x: x if x < 0 else 0)
            loss_aversion_intensity = (downward_movement * self.data['volume']) * loss_aversion_factor
            self.data[f'loss_aversion_{day}'] = loss_aversion_intensity
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha59(self, days=list, confirmation_bias_factor=0.4):
        """
        The alpha is based on confirmation bias, where traders seek information that confirms their beliefs.
        We use volume during uptrends or downtrends to measure the confirmation bias effect.

        Parameter:
        - days (list): the time intervals for calculating confirmation bias
        - confirmation_bias_factor (float): factor to scale confirmation bias impact (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_change = self.data['close'].pct_change(day)
            volume_impact = self.data['volume']
            self.data[f'confirmation_bias_{day}'] = (price_change * volume_impact) * confirmation_bias_factor
        self.data = self.data.copy() 
        return self.data

    def alpha60(self, days=list, exhaustion_risk_factor=0.35):
        """
        The alpha is based on the exhaustion risk, where traders get fatigued after extended periods of high volume.
        We measure this by calculating extended high volume periods relative to price stagnation.

        Parameter:
        - days (list): the time intervals for calculating exhaustion risk
        - exhaustion_risk_factor (float): factor to scale exhaustion risk impact (default is 0.35)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            high_volume = self.data['volume'] > self.data['volume'].rolling(window=day).mean()
            price_stagnation = self.data['close'].pct_change(day).apply(lambda x: 1 if abs(x) < 0.01 else 0)
            exhaustion_risk = (high_volume * price_stagnation) * exhaustion_risk_factor
            self.data[f'exhaustion_risk_{day}'] = exhaustion_risk
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data
    
    def alpha61(self, days=list, time_lag_factor=0.6):
        """
        The alpha is based on time lag effect, where traders react with a delay to price changes.
        We use a lagged volume to evaluate its impact on current price changes.

        Parameter:
        - days (list): the time intervals for calculating time lag effect
        - time_lag_factor (float): factor to scale the time lag impact (default is 0.6)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            lagged_volume = self.data['volume'].shift(day)
            price_change = self.data['close'].pct_change(day)
            self.data[f'time_lag_{day}'] = (lagged_volume * price_change) * time_lag_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha62(self, days=list, delayed_reaction_factor=0.45):
        """
        The alpha is based on delayed market reaction, where traders do not react immediately to price movements.
        We calculate the effect of past price changes on current volume.

        Parameter:
        - days (list): the time intervals for calculating delayed reaction effect
        - delayed_reaction_factor (float): factor to scale delayed reaction impact (default is 0.45)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            past_price_change = self.data['close'].pct_change(day).shift(day)
            self.data[f'delayed_reaction_{day}'] = (past_price_change * self.data['volume']) * delayed_reaction_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha63(self, days=list, memory_decay_factor=0.3):
        """
        The alpha is based on memory decay, where the impact of past events fades over time.
        We use an exponentially weighted moving average to model memory decay on volume and price.

        Parameter:
        - days (list): the time intervals for calculating memory decay effect
        - memory_decay_factor (float): factor to scale memory decay impact (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            ewm_volume = self.data['volume'].ewm(span=day).mean()
            ewm_price_change = self.data['close'].ewm(span=day).mean()
            self.data[f'memory_decay_{day}'] = (ewm_volume * ewm_price_change) * memory_decay_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha64(self, days=list, temporal_discrepancy_factor=0.5):
        """
        The alpha is based on temporal discrepancy, where traders perceive value differently over time.
        We calculate discrepancies between past and present price movements using volume as a weight.

        Parameter:
        - days (list): the time intervals for calculating temporal discrepancy
        - temporal_discrepancy_factor (float): factor to scale temporal discrepancy impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            past_price = self.data['close'].shift(day)
            current_price = self.data['close']
            volume_weight = self.data['volume'].rolling(window=day).mean()
            self.data[f'temporal_discrepancy_{day}'] = ((current_price - past_price) * volume_weight) * temporal_discrepancy_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha65(self, days=list, reaction_lag_factor=0.4):
        """
        The alpha is based on reaction lag, where traders take time to react to new information.
        We assess the impact of lagged price movements on current volume changes.

        Parameter:
        - days (list): the time intervals for calculating reaction lag effect
        - reaction_lag_factor (float): factor to scale reaction lag impact (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            lagged_price_change = self.data['close'].pct_change(day).shift(day)
            volume_change = self.data['volume'].pct_change(day)
            self.data[f'reaction_lag_{day}'] = (lagged_price_change * volume_change) * reaction_lag_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha66(self, days=list, volatility_memory_factor=0.55):
        """
        The alpha is based on volatility memory, where traders remember recent high volatility periods.
        We use past volatility to estimate its effect on current volume and price.

        Parameter:
        - days (list): the time intervals for calculating volatility memory effect
        - volatility_memory_factor (float): factor to scale volatility memory impact (default is 0.55)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            past_volatility = self.data['close'].rolling(window=day).std().shift(day)
            current_volume = self.data['volume']
            self.data[f'volatility_memory_{day}'] = (past_volatility * current_volume) * volatility_memory_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha67(self, days=list, anticipation_effect_factor=0.5):
        """
        改良後的 Alpha67：針對最新數據適配，避免因滾動窗口不足導致數據無法使用。

        Parameters:
        - days (list): 時間間隔 (分鐘) 用於計算滾動特徵
        - anticipation_effect_factor (float): 因子，用於縮放預期影響（默認為0.5）

        Returns:
        - df (pd.DataFrame): 返回包含特徵的數據
        """
        for day in days:
            # 滾動窗口的歷史百分比變化（適配邊界情況）
            historical_price_change = (
                self.data["close"]
                .pct_change(day)
                .rolling(window=min(day, len(self.data)))  # 使用可用窗口長度
                .mean()
            )
            
            # 如果窗口不足，用過去的均值或插值估算
            historical_price_change.fillna(historical_price_change.mean(), inplace=True)
            
            current_volume = self.data["volume"]
            
            # 預期影響特徵計算
            self.data[f"anticipation_effect_{day}"] = (
                historical_price_change * current_volume
            ) * anticipation_effect_factor

        # 防止DataFrame內存碎片化
        self.data = self.data.copy()
        return self.data


    def alpha68(self, days=list, lagged_volatility_factor=0.35):
        """
        The alpha is based on lagged volatility, where traders react to previous periods of volatility.
        We calculate the impact of past volatility on current trading volume.

        Parameter:
        - days (list): the time intervals for calculating lagged volatility effect
        - lagged_volatility_factor (float): factor to scale lagged volatility impact (default is 0.35)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            past_volatility = self.data['close'].rolling(window=day).std().shift(day)
            self.data[f'lagged_volatility_{day}'] = past_volatility * self.data['volume'] * lagged_volatility_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha69(self, days=list, lagged_volume_price_factor=0.65):
        """
        The alpha is based on the interaction of lagged volume and price changes.
        We assess the effect of past volume on the current price trend.

        Parameter:
        - days (list): the time intervals for calculating lagged volume effect
        - lagged_volume_price_factor (float): factor to scale lagged volume price impact (default is 0.65)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            lagged_volume = self.data['volume'].shift(day)
            price_change = self.data['close'].pct_change(day)
            self.data[f'lagged_volume_price_{day}'] = (lagged_volume * price_change) * lagged_volume_price_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha70(self, days=list, persistence_factor=0.7):
        """
        The alpha is based on persistence, where traders' behavior persists over time due to habit formation.
        We use lagged values of volume and price changes to model this persistence.

        Parameter:
        - days (list): the time intervals for calculating persistence effect
        - persistence_factor (float): factor to scale persistence impact (default is 0.7)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            lagged_price_change = self.data['close'].pct_change(day).shift(day)
            lagged_volume = self.data['volume'].shift(day)
            self.data[f'persistence_{day}'] = (lagged_price_change * lagged_volume) * persistence_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha71(self, days=list, rsi_period=14, divergence_factor=0.5):
        """
        The alpha is based on bullish or bearish divergence between RSI and price.
        We calculate RSI and identify divergences where price is making new highs/lows but RSI is not.

        Parameter:
        - days (list): the time intervals for calculating divergence effect
        - rsi_period (int): the number of periods for calculating RSI (default is 14)
        - divergence_factor (float): factor to scale divergence impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        delta = self.data['close'].diff(1)
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        for day in days:
            price_diff = self.data['close'].diff(day)
            rsi_diff = rsi.diff(day)
            self.data[f'bull_bear_divergence_{day}'] = (price_diff * -rsi_diff) * divergence_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha72(self, days=list, macd_short=12, macd_long=26, signal_smooth=9, cross_factor=0.4):
        """
        The alpha is based on MACD crossovers as a turning point indicator.
        We compute the MACD line and signal line, and derive alpha from crossovers.

        Parameter:
        - days (list): the time intervals for calculating MACD effect
        - macd_short (int): short period for MACD calculation (default is 12)
        - macd_long (int): long period for MACD calculation (default is 26)
        - signal_smooth (int): period for MACD signal line smoothing (default is 9)
        - cross_factor (float): factor to scale the MACD cross impact (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        short_ema = self.data['close'].ewm(span=macd_short, min_periods=1).mean()
        long_ema = self.data['close'].ewm(span=macd_long, min_periods=1).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_smooth, min_periods=1).mean()
        macd_diff = macd - signal
        
        for day in days:
            self.data[f'macd_crossover_{day}'] = macd_diff.shift(day) * cross_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha73(self, days=list, stochastic_k=14, stochastic_d=3, stochastic_factor=0.3):
        """
        The alpha is based on the stochastic oscillator, focusing on overbought and oversold levels.
        We calculate %K and %D lines and determine turning points when %K crosses %D.

        Parameter:
        - days (list): the time intervals for calculating stochastic effect
        - stochastic_k (int): the period for %K calculation (default is 14)
        - stochastic_d (int): the period for %D smoothing (default is 3)
        - stochastic_factor (float): factor to scale stochastic oscillator impact (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        low_min = self.data['low'].rolling(window=stochastic_k).min()
        high_max = self.data['high'].rolling(window=stochastic_k).max()
        percent_k = 100 * ((self.data['close'] - low_min) / (high_max - low_min))
        percent_d = percent_k.rolling(window=stochastic_d).mean()
        
        for day in days:
            self.data[f'stochastic_turn_{day}'] = (percent_k - percent_d).shift(day) * stochastic_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha74(self, days=list, bollinger_period=20, num_std_dev=2, bb_factor=0.35):
        """
        The alpha is based on Bollinger Bands, where price touches or crosses bands, signaling a reversal.
        We use the width between upper and lower Bollinger Bands to assess market volatility and reversals.

        Parameter:
        - days (list): the time intervals for calculating Bollinger Band effect
        - bollinger_period (int): period for calculating the Bollinger Bands (default is 20)
        - num_std_dev (int): number of standard deviations for the bands (default is 2)
        - bb_factor (float): factor to scale Bollinger Band impact (default is 0.35)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        sma = self.data['close'].rolling(window=bollinger_period).mean()
        rolling_std = self.data['close'].rolling(window=bollinger_period).std()
        upper_band = sma + (rolling_std * num_std_dev)
        lower_band = sma - (rolling_std * num_std_dev)
        
        for day in days:
            price_position = (self.data['close'] - lower_band) / (upper_band - lower_band)
            self.data[f'bollinger_reversal_{day}'] = price_position.shift(day) * bb_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha75(self, days=list, adx_period=14, adx_factor=0.5):
        """
        The alpha is based on the Average Directional Index (ADX), which measures trend strength.
        Turning points are identified when ADX peaks or troughs, signaling potential trend reversals.

        Parameter:
        - days (list): the time intervals for calculating ADX effect
        - adx_period (int): period for ADX calculation (default is 14)
        - adx_factor (float): factor to scale ADX impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        high_diff = self.data['high'].diff()
        low_diff = self.data['low'].diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        tr = np.max([self.data['high'] - self.data['low'],
                     abs(self.data['high'] - self.data['close'].shift()),
                     abs(self.data['low'] - self.data['close'].shift())], axis=0)
        atr = pd.Series(tr).rolling(window=adx_period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=adx_period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=adx_period).mean() / atr)
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=adx_period).mean()
        
        for day in days:
            self.data[f'adx_trend_strength_{day}'] = adx.shift(day) * adx_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha76(self, days=list, volume_momentum_factor=0.25):
        """
        The alpha is based on volume momentum, where spikes in volume indicate potential turning points.
        We calculate the rate of change in volume and use it as a proxy for sentiment shifts.

        Parameter:
        - days (list): the time intervals for calculating volume momentum
        - volume_momentum_factor (float): factor to scale volume momentum impact (default is 0.25)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            volume_change = self.data['volume'].pct_change(day)
            self.data[f'volume_momentum_{day}'] = volume_change * volume_momentum_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha77(self, days=list, volume_weighted_factor=0.3):
        """
        The alpha is based on volume-weighted average price (VWAP) and volume changes.
        We calculate the VWAP and assess deviations to identify significant price moves.

        Parameter:
        - days (list): the time intervals for calculating VWAP effect
        - volume_weighted_factor (float): factor to scale VWAP impact (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            vwap = (self.data['close'] * self.data['volume']).rolling(window=day).sum() / self.data['volume'].rolling(window=day).sum()
            self.data[f'vwap_deviation_{day}'] = (self.data['close'] - vwap) * volume_weighted_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha78(self, days=list, volume_price_trend_factor=0.2):
        """
        The alpha is based on Volume Price Trend (VPT), which combines price changes and volume to identify shifts in trend.
        We calculate VPT and apply a scaling factor to assess the strength of price movements.

        Parameter:
        - days (list): the time intervals for calculating VPT effect
        - volume_price_trend_factor (float): factor to scale VPT impact (default is 0.2)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        vpt = (self.data['close'].pct_change() * self.data['volume']).cumsum()
        for day in days:
            self.data[f'volume_price_trend_{day}'] = vpt.shift(day) * volume_price_trend_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha79(self, days=list, on_balance_volume_factor=0.15):
        """
        The alpha is based on On-Balance Volume (OBV), which uses volume flow to predict changes in stock price.
        We calculate OBV and apply a scaling factor to estimate the impact of volume on price movements.

        Parameter:
        - days (list): the time intervals for calculating OBV effect
        - on_balance_volume_factor (float): factor to scale OBV impact (default is 0.15)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        obv = (np.sign(self.data['close'].diff()) * self.data['volume']).cumsum()
        for day in days:
            self.data[f'on_balance_volume_{day}'] = obv.shift(day) * on_balance_volume_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha80(self, days=list, volume_breakout_factor=0.4):
        """
        The alpha is based on volume breakouts, where significant changes in volume indicate potential trend reversals.
        We identify breakouts by calculating volume z-scores and apply a scaling factor to assess their impact.

        Parameter:
        - days (list): the time intervals for calculating volume breakout effect
        - volume_breakout_factor (float): factor to scale volume breakout impact (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        volume_mean = self.data['volume'].rolling(window=20).mean()
        volume_std = self.data['volume'].rolling(window=20).std()
        volume_zscore = (self.data['volume'] - volume_mean) / volume_std
        for day in days:
            self.data[f'volume_breakout_{day}'] = volume_zscore.shift(day) * volume_breakout_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha81(self, days=list, volume_lag_factor=0.35):
        """
        The alpha is derived from the concept of timelag between volume changes and price reaction.
        We apply a lagged correlation between volume and close price to detect delayed reactions.

        Parameter:
        - days (list): the time intervals for calculating timelag effect
        - volume_lag_factor (float): factor to scale the timelag impact (default is 0.35)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            volume_lag = self.data['volume'].shift(day)
            self.data[f'volume_timelag_{day}'] = ((self.data['close'] - volume_lag) * volume_lag_factor)
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha82(self, days=list, emotional_volume_shift_factor=0.45):
        """
        The alpha is based on emotional volume shifts, representing periods of heightened trading activity.
        We use exponential weighted volume changes to detect emotional surges in trading.

        Parameter:
        - days (list): the time intervals for calculating emotional volume shifts
        - emotional_volume_shift_factor (float): factor to scale the emotional impact (default is 0.45)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            ewm_volume = self.data['volume'].ewm(span=day).mean()
            self.data[f'emotional_volume_shift_{day}'] = ewm_volume * emotional_volume_shift_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha83(self, days=list, delayed_reaction_factor=0.5):
        """
        The alpha is derived from the theory that volume surges lead to delayed price reactions.
        We calculate the delayed impact of volume spikes on close price using a scaling factor.

        Parameter:
        - days (list): the time intervals for calculating delayed reaction effect
        - delayed_reaction_factor (float): factor to scale delayed reaction impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            volume_spike = self.data['volume'].rolling(window=day).max()
            self.data[f'delayed_reaction_{day}'] = (self.data['close'] - volume_spike) * delayed_reaction_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha84(self, days=list, regret_aversion_factor=0.3):
        """
        The alpha is based on regret aversion theory, where traders avoid making decisions that could lead to regret.
        We assess the average volume traded during periods of price declines to quantify regret aversion.

        Parameter:
        - days (list): the time intervals for calculating regret aversion effect
        - regret_aversion_factor (float): factor to scale regret aversion impact (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_decline = self.data['close'].diff() < 0
            regret_volume = self.data['volume'][price_decline].rolling(window=day).mean()
            self.data[f'regret_aversion_{day}'] = regret_volume * regret_aversion_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha85(self, days=list, endowment_effect_factor=0.25):
        """
        The alpha is based on the endowment effect, where traders value assets they own more highly.
        We calculate the volume-weighted change in close price to model perceived overvaluation.

        Parameter:
        - days (list): the time intervals for calculating endowment effect
        - endowment_effect_factor (float): factor to scale endowment effect impact (default is 0.25)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            weighted_price_change = (self.data['close'].diff() * self.data['volume']).rolling(window=day).mean()
            self.data[f'endowment_effect_{day}'] = weighted_price_change * endowment_effect_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha86(self, days=list, fibonacci_volume_factor=0.15):
        """
        The alpha is derived from using the Fibonacci sequence to identify points of interest in trading activity.
        We apply the Fibonacci factor to volume changes to highlight specific retracement levels.

        Parameter:
        - days (list): the time intervals for calculating Fibonacci volume effect
        - fibonacci_volume_factor (float): factor to scale volume impact using Fibonacci (default is 0.15)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        golden_ratio = 1.618
        for day in days:
            fibonacci_volume = self.data['volume'].rolling(window=int(day * golden_ratio)).mean()
            self.data[f'fibonacci_volume_{day}'] = fibonacci_volume * fibonacci_volume_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha87(self, days=list, golden_ratio_price_factor=0.4):
        """
        The alpha incorporates the golden ratio to evaluate price levels that may act as psychological support or resistance.
        We calculate a weighted average price with a golden ratio factor to determine these levels.

        Parameter:
        - days (list): the time intervals for calculating golden ratio price effect
        - golden_ratio_price_factor (float): factor to scale price impact using the golden ratio (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        golden_ratio = 1.618
        for day in days:
            weighted_price = (self.data['close'] * golden_ratio + self.data['open']) / (1 + golden_ratio)
            self.data[f'golden_ratio_price_{day}'] = weighted_price.rolling(window=day).mean() * golden_ratio_price_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha88(self, days=list, golden_volume_divergence_factor=0.2):
        """
        The alpha measures divergence between price and volume using the golden ratio.
        We apply the golden ratio to detect significant divergences that could signal reversals.

        Parameter:
        - days (list): the time intervals for calculating divergence effect
        - golden_volume_divergence_factor (float): factor to scale divergence impact (default is 0.2)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        golden_ratio = 1.618
        for day in days:
            price_volume_diff = (self.data['close'] - self.data['volume'].rolling(window=day).mean()) * golden_ratio
            self.data[f'golden_volume_divergence_{day}'] = price_volume_diff * golden_volume_divergence_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha89(self, days=list, golden_cross_emotion_factor=0.5):
        """
        The alpha is based on the golden cross concept, where a short-term moving average crosses above a long-term one.
        We apply the golden ratio to capture emotional responses to golden cross signals.

        Parameter:
        - days (list): the time intervals for calculating golden cross effect
        - golden_cross_emotion_factor (float): factor to scale emotional reaction to golden cross (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        golden_ratio = 1.618
        for day in days:
            short_ma = self.data['close'].rolling(window=int(day / golden_ratio)).mean()
            long_ma = self.data['close'].rolling(window=int(day * golden_ratio)).mean()
            self.data[f'golden_cross_emotion_{day}'] = (short_ma - long_ma) * golden_cross_emotion_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha90(self, days=list, retracement_anticipation_factor=0.35):
        """
        The alpha is derived from Fibonacci retracement, where price levels are expected to retrace by certain ratios.
        We anticipate price retracement levels and quantify the effect using a scaling factor.

        Parameter:
        - days (list): the time intervals for calculating retracement effect
        - retracement_anticipation_factor (float): factor to scale retracement anticipation impact (default is 0.35)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        golden_ratio = 1.618
        for day in days:
            retracement_level = self.data['close'].rolling(window=day).apply(lambda x: (x.max() - x.min()) * golden_ratio)
            self.data[f'retracement_anticipation_{day}'] = retracement_level * retracement_anticipation_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha91(self, days=list, retracement_momentum_factor=0.25):
        """
        The alpha is derived from combining Fibonacci retracement concepts with momentum analysis.
        We apply a retracement factor to calculate the momentum of price changes.

        Parameter:
        - days (list): the time intervals for calculating retracement momentum effect
        - retracement_momentum_factor (float): factor to scale the momentum impact using retracement (default is 0.25)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            momentum = self.data['close'].diff(periods=day)
            self.data[f'retracement_momentum_{day}'] = momentum * retracement_momentum_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha92(self, days=list, retracement_fear_greed_factor=0.4):
        """
        The alpha is based on Fibonacci retracement to quantify fear and greed in the market.
        We use the ratio between high and low prices, scaled by a retracement factor, to assess emotional extremes.

        Parameter:
        - days (list): the time intervals for calculating fear and greed effect
        - retracement_fear_greed_factor (float): factor to scale fear and greed impact (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            fear_greed = ((self.data['high'] - self.data['low']) / self.data['low']).rolling(window=day).mean()
            self.data[f'retracement_fear_greed_{day}'] = fear_greed * retracement_fear_greed_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha93(self, days=list, retracement_vix_adjustment_factor=0.3):
        """
        The alpha adjusts price movements by incorporating retracement levels and market volatility (VIX).
        We calculate the adjusted close price by scaling with a retracement factor to represent market sentiment.

        Parameter:
        - days (list): the time intervals for calculating VIX adjustment effect
        - retracement_vix_adjustment_factor (float): factor to scale the VIX adjustment impact (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            adjusted_close = (self.data['close']).rolling(window=day).mean()
            self.data[f'retracement_vix_adjustment_{day}'] = adjusted_close * retracement_vix_adjustment_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha94(self, days=list, retracement_price_reversal_factor=0.5):
        """
        The alpha aims to capture price reversal points based on Fibonacci retracement levels.
        We use the high-low range combined with a retracement factor to predict potential reversal zones.

        Parameter:
        - days (list): the time intervals for calculating price reversal effect
        - retracement_price_reversal_factor (float): factor to scale price reversal impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            price_reversal = (self.data['high'] - self.data['low']).rolling(window=day).mean()
            self.data[f'retracement_price_reversal_{day}'] = price_reversal * retracement_price_reversal_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha95(self, days=list, retracement_emotional_intensity_factor=0.45):
        """
        The alpha is based on emotional intensity theory, incorporating retracement levels to assess trading behavior.
        We calculate the rate of change in close price and multiply by an emotional intensity factor derived from retracement.

        Parameter:
        - days (list): the time intervals for calculating emotional intensity
        - retracement_emotional_intensity_factor (float): factor to scale emotional intensity impact (default is 0.45)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            rate_of_change = self.data['close'].pct_change(day)
            self.data[f'retracement_emotional_intensity_{day}'] = rate_of_change * retracement_emotional_intensity_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha96(self, days=list, retracement_support_resistance_factor=0.35):
        """
        The alpha identifies support and resistance levels based on Fibonacci retracement.
        We calculate moving averages of close prices and scale them using retracement to identify key levels.

        Parameter:
        - days (list): the time intervals for calculating support and resistance
        - retracement_support_resistance_factor (float): factor to scale support and resistance impact (default is 0.35)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            support_resistance = self.data['close'].rolling(window=day).mean()
            self.data[f'retracement_support_resistance_{day}'] = support_resistance * retracement_support_resistance_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha97(self, days=list, retracement_volatility_factor=0.4):
        """
        The alpha leverages retracement concepts to evaluate market volatility.
        We apply a retracement factor to calculate volatility of the close price over different time windows.

        Parameter:
        - days (list): the time intervals for calculating volatility
        - retracement_volatility_factor (float): factor to scale volatility impact (default is 0.4)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            volatility = self.data['close'].rolling(window=day).std()
            self.data[f'retracement_volatility_{day}'] = volatility * retracement_volatility_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha98(self, days=list, retracement_peak_trough_factor=0.5):
        """
        The alpha identifies peaks and troughs in price movements using retracement levels.
        We calculate the difference between rolling highs and lows.

        Parameter:
        - days (list): the time intervals for calculating peaks and troughs
        - retracement_peak_trough_factor (float): factor to scale peak and trough impact (default is 0.5)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            peak_trough_diff = (self.data['high'].rolling(window=day).max() - self.data['low'].rolling(window=day).min())
            self.data[f'retracement_peak_trough_{day}'] = peak_trough_diff * retracement_peak_trough_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha99(self, days=list, retracement_confluence_factor=1):
        """
        The alpha measures confluence between retracement levels and price momentum.
        We identify periods where both price retracement and momentum align, using a retracement factor to quantify the confluence.

        Parameter:
        - days (list): the time intervals for calculating confluence
        - retracement_confluence_factor (float): factor to scale confluence impact (default is 0.6)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            retracement = (self.data['close'].rolling(window=day).max() - self.data['close'].rolling(window=day).min())
            momentum = self.data['close'].diff(periods=day)
            self.data[f'retracement_confluence_{day}'] = (retracement * momentum) * retracement_confluence_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha100(self, days=list, retracement_volume_intensity_factor=0.5):
        """
        The alpha evaluates the intensity of volume movements in conjunction with retracement levels.
        We apply the retracement factor to detect surges in volume and their potential impact on price changes.

        Parameter:
        - days (list): the time intervals for calculating volume intensity
        - retracement_volume_intensity_factor (float): factor to scale volume intensity impact (default is 0.3)

        Output:
        - df (pd.Dataframe): return the input df with alpha series
        """
        for day in days:
            volume_intensity = self.data['volume'].rolling(window=day).sum()
            self.data[f'retracement_volume_intensity_{day}'] = volume_intensity * retracement_volume_intensity_factor
        self.data = self.data.copy()  # Defragment the DataFrame to improve performance
        return self.data

    def alpha101(self, days=list):
        """
        Alpha to identify if the market is in a short-term bearish phase using moving averages.
        Parameter:
        - days (list): the time intervals for calculating moving averages
        """
        for day in days:
            ma = self.data['close'].rolling(window=day).mean()
            self.data[f'short_term_bearish_{day}'] = np.where(self.data['close'] < ma, 1, 0)
        return self.data

    def alpha102(self, days=list):
        """
        Alpha to identify if the market is in a medium-term bullish phase using moving averages.
        Parameter:
        - days (list): the time intervals for calculating moving averages
        """
        for day in days:
            ma = self.data['close'].rolling(window=day).mean()
            self.data[f'medium_term_bullish_{day}'] = np.where(self.data['close'] > ma, 1, 0)
        return self.data

    def alpha103(self, days=list, threshold=1.5):
        """
        Alpha to assess if the market is in a high volatility phase, indicating potential trend changes.
        Parameter:
        - days (list): the time intervals for calculating volatility
        - threshold (float): the threshold to determine high volatility
        """
        for day in days:
            rolling_std = self.data['close'].rolling(window=day).std()
            self.data[f'high_volatility_{day}'] = np.where(rolling_std > threshold, 1, 0)
        return self.data

    def alpha104(self, days=list):
        """
        Alpha to determine if the market is in a consolidation phase using Bollinger Bands.
        Parameter:
        - days (list): the time intervals for calculating Bollinger Bands
        """
        for day in days:
            ma = self.data['close'].rolling(window=day).mean()
            std = self.data['close'].rolling(window=day).std()
            upper_band = ma + (2 * std)
            lower_band = ma - (2 * std)
            self.data[f'consolidation_{day}'] = np.where((self.data['close'] < upper_band) & (self.data['close'] > lower_band), 1, 0)
        return self.data

    def alpha105(self, days=list, trend_factor=0.05):
        """
        Alpha to evaluate if the market is in a trending phase based on the Average Directional Index (ADX).
        Parameter:
        - days (list): the time intervals for calculating ADX
        - trend_factor (float): the factor to determine a strong trend
        """
        for day in days:
            high_low_diff = self.data['high'] - self.data['low']
            adx = high_low_diff.rolling(window=day).mean() / self.data['close'] * 100
            self.data[f'trending_phase_{day}'] = np.where(adx > trend_factor, 1, 0)
        return self.data

    def alpha106(self, days=list):
        """
        Alpha to identify bear market conditions using Exponential Moving Averages (EMA).
        Parameter:
        - days (list): the time intervals for calculating EMAs
        """
        for day in days:
            ema = self.data['close'].ewm(span=day, adjust=False).mean()
            self.data[f'bear_market_{day}'] = np.where(self.data['close'] < ema, 1, 0)
        return self.data

    def alpha107(self, days=list):
        """
        Alpha to identify bull market conditions using Exponential Moving Averages (EMA).
        Parameter:
        - days (list): the time intervals for calculating EMAs
        """
        for day in days:
            ema = self.data['close'].ewm(span=day, adjust=False).mean()
            self.data[f'bull_market_{day}'] = np.where(self.data['close'] > ema, 1, 0)
        return self.data

    def alpha108(self, days=list, rsi_threshold=80):
        """
        Alpha to detect overbought market conditions using Relative Strength Index (RSI).
        Parameter:
        - days (list): the time intervals for calculating RSI
        - rsi_threshold (int): the threshold to determine overbought conditions (default is 70)
        """
        for day in days:
            delta = self.data['close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=day).mean()
            avg_loss = loss.rolling(window=day).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            self.data[f'overbought_rsi_{day}'] = np.where(rsi > rsi_threshold, 1, 0)
        return self.data

    def alpha109(self, days=list, rsi_threshold=20):
        """
        Alpha to detect oversold market conditions using Relative Strength Index (RSI).
        Parameter:
        - days (list): the time intervals for calculating RSI
        - rsi_threshold (int): the threshold to determine oversold conditions (default is 30)
        """
        for day in days:
            delta = self.data['close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=day).mean()
            avg_loss = loss.rolling(window=day).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            self.data[f'oversold_rsi_{day}'] = np.where(rsi < rsi_threshold, 1, 0)
        return self.data

    def alpha110(self, days=list):
        """
        Alpha to identify potential reversal phases using Moving Average Convergence Divergence (MACD).
        Parameter:
        - days (list): the time intervals for calculating MACD
        """
        for day in days:
            ema_short = self.data['close'].ewm(span=day, adjust=False).mean()
            ema_long = self.data['close'].ewm(span=day * 2, adjust=False).mean()
            macd = ema_short - ema_long
            signal = macd.ewm(span=day // 2, adjust=False).mean()
            self.data[f'macd_reversal_{day}'] = np.where(macd > signal, 1, 0)
        return self.data    


    def alpha111(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that uses Bollinger Bands, volume moving average, and price range.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            # 获取数据
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算收盘价的简单移动平均（SMA）
            sma_close = close.rolling(window=day).mean()
            
            # 计算布林带上下轨
            rolling_std = close.rolling(window=day).std()
            upper_band = sma_close + 2 * rolling_std
            lower_band = sma_close - 2 * rolling_std
            bb_percent = (close - lower_band) / (upper_band - lower_band)
            
            # 计算成交量的移动平均
            volume_mean = volume.rolling(window=day).mean()
            
            # 计算价格变动（高低差）
            price_range = high - low
            
            # 计算alpha因子
            alpha = (bb_percent * volume_mean * price_range) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha111_{day}'] = alpha
        return self.data




    def alpha112(self, days=[10, 20], volume_window=15, bb_window=20, anticipation_effect_factor=0.6):
        """
        Alpha112：綜合多個技術指標來捕捉價格變動趨勢。
        
        參數:
        - days (list): 用於計算不同時間窗口的時間間隔。
        - volume_window (int): 成交量的滾動窗口週期。
        - bb_window (int): 布林帶的滾動窗口週期。
        - anticipation_effect_factor (float): 因子，用於縮放預期影響（默認為0.6）。
        
        返回:
        - df (pd.DataFrame): 返回包含新特徵的數據。
        """
        for day in days:
            # 計算短期和長期指數移動平均線 (EMA)
            ema_short = self.data['close'].ewm(span=day, min_periods=1).mean()
            ema_long = self.data['close'].ewm(span=day*2, min_periods=1).mean()  # 假設長期為短期的兩倍
            macd = ema_short - ema_long

            # 計算布林帶上下界
            rolling_mean = self.data['close'].rolling(window=bb_window).mean()
            rolling_std = self.data['close'].rolling(window=bb_window).std()
            bb_upper = rolling_mean + (2 * rolling_std)
            bb_lower = rolling_mean - (2 * rolling_std)
            bb_width = bb_upper - bb_lower

            # 計算成交量的滾動平均值
            volume_mean = self.data['volume'].rolling(window=volume_window).mean()

            # 綜合計算：考慮價格變動的綜合因子
            self.data[f'alpha112_{day}'] = (
                macd * bb_width * volume_mean
            ) * anticipation_effect_factor

        # 防止DataFrame內存碎片化
        self.data = self.data.copy()
        return self.data

    def alpha113(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that uses price range and volume moving average.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            # 获取数据
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算价格变动（高低差）
            price_range = high - low
            
            # 计算收盘价与高低差的比例
            price_ratio = close / price_range
            
            # 计算成交量的移动平均
            volume_mean = volume.rolling(window=day).mean()
            
            # 计算alpha因子
            alpha = (price_ratio * volume_mean) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha113_{day}'] = alpha
        return self.data


    def alpha114(self, days=[5, 10, 20], volume_window=10, std_window=20, anticipation_effect_factor=0.8):
        """
        Alpha114：綜合布林帶、成交量和價格波動的修正因子。
        
        參數:
        - days (list): 用於計算不同時間窗口的時間間隔。
        - volume_window (int): 成交量的滾動窗口週期。
        - std_window (int): 用於計算價格標準差的滾動窗口。
        - anticipation_effect_factor (float): 因子，用於縮放預期影響（默認為0.8）。
        
        返回:
        - df (pd.DataFrame): 返回包含新特徵的數據。
        """
        for day in days:
            # 計算布林帶百分比
            rolling_mean = self.data['close'].rolling(window=day).mean()
            rolling_std = self.data['close'].rolling(window=std_window).std()
            bb_upper = rolling_mean + (2 * rolling_std)
            bb_lower = rolling_mean - (2 * rolling_std)
            bb_percent = (self.data['close'] - bb_lower) / (bb_upper - bb_lower)

            # 計算成交量的滾動平均值
            volume_mean = self.data['volume'].rolling(window=volume_window).mean()

            # 綜合計算：布林帶和成交量的影響
            self.data[f'alpha114_{day}'] = (
                bb_percent * volume_mean
            ) * anticipation_effect_factor

        # 防止DataFrame內存碎片化
        self.data = self.data.copy()
        return self.data

    def alpha115(self, days=[10, 20], volume_window=10, tscore_window=30, anticipation_effect_factor=0.6):
        """
        Alpha115：結合價格波動（t分數）和成交量，預測價格跳空。
        
        參數:
        - days (list): 用於計算不同時間窗口的時間間隔。
        - volume_window (int): 成交量的滾動窗口週期。
        - tscore_window (int): 用於計算 t 分數的滾動窗口週期。
        - anticipation_effect_factor (float): 因子，用於縮放預期影響（默認為0.6）。
        
        返回:
        - df (pd.DataFrame): 返回包含新特徵的數據。
        """
        for day in days:
            # 計算滾動的t分數
            rolling_mean = self.data['close'].rolling(window=tscore_window).mean()
            rolling_std = self.data['close'].rolling(window=tscore_window).std()
            tscore = (self.data['close'] - rolling_mean) / rolling_std

            # 計算成交量的滾動平均值
            volume_mean = self.data['volume'].rolling(window=volume_window).mean()

            # 綜合計算：t分數與成交量的影響
            self.data[f'alpha115_{day}'] = (
                tscore * volume_mean
            ) * anticipation_effect_factor

        # 防止DataFrame內存碎片化
        self.data = self.data.copy()
        return self.data   

    def alpha116(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that calculates price range weighted by volume.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            high = self.data['high']
            low = self.data['low']
            volume = self.data['volume']
            
            # 计算价格变动（高低差）
            price_range = high - low
            
            # 计算成交量的移动平均
            volume_mean = volume.rolling(window=day).mean()
            
            # 计算alpha116：成交量加权的价格变动
            alpha = (price_range * volume_mean) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha116_{day}'] = alpha
        return self.data


    def alpha117(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that uses price momentum and volume.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算价格动量（收盘价的差异）
            price_momentum = close.diff(day)
            
            # 计算成交量的移动平均
            volume_mean = volume.rolling(window=day).mean()
            
            # 计算alpha117：价格动量与成交量的加权组合
            alpha = (price_momentum * volume_mean) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha117_{day}'] = alpha
        return self.data

    def alpha118(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator using Bollinger Bands and volume with adjusted weights.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算收盘价的简单移动平均
            sma_close = close.rolling(window=day).mean()
            
            # 计算布林带的上下轨
            rolling_std = close.rolling(window=day).std()
            upper_band = sma_close + 2 * rolling_std
            lower_band = sma_close - 2 * rolling_std
            bb_percent = (close - lower_band) / (upper_band - lower_band)
            
            # 计算成交量的移动平均
            volume_mean = volume.rolling(window=day).mean()
            
            # 计算alpha118：布林带和成交量的加权组合
            alpha = (bb_percent * volume_mean) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha118_{day}'] = alpha
        return self.data


    def alpha119(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator using price momentum and volume trend.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算价格动量（高低差的移动平均）
            price_momentum = (high - low).rolling(window=day).mean()
            
            # 计算成交量的移动平均
            volume_mean = volume.rolling(window=day).mean()
            
            # 计算alpha119：价格动量与成交量趋势的加权组合
            alpha = (price_momentum * volume_mean) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha119_{day}'] = alpha
        return self.data


    def alpha120(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator combining price range and volume momentum.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算价格波动（高低差）
            price_range = high - low
            
            # 计算成交量的动量（成交量差异）
            volume_momentum = volume.diff(day)
            
            # 计算alpha120：价格波动与成交量动量的加权组合
            alpha = (price_range * volume_momentum) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha120_{day}'] = alpha
        return self.data

    def alpha121(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that detects price gaps and calculates the Z-score for price change after gap.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            close = self.data['close']
            
            # 计算跳空：开盘与前一日收盘价的差值
            gap = close.diff(1)
            
            # 计算价格变化的 Z 分数
            mean_gap = gap.rolling(window=day).mean()
            std_gap = gap.rolling(window=day).std()
            z_score_gap = (gap - mean_gap) / std_gap
            
            # 计算 alpha121：跳空的 Z 分数加权
            alpha = z_score_gap * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha121_{day}'] = alpha
        return self.data

    def alpha122(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that calculates the T-score for price gaps and pullback movements.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            close = self.data['close']
            
            # 计算跳空：开盘与前一日收盘价的差值
            gap = close.diff(1)
            
            # 计算价格回撤：当前价格与最大涨幅的比值
            max_gap = gap.rolling(window=day).max()
            pullback = gap / max_gap
            
            # 计算 T 分数
            t_stat = (pullback - pullback.mean()) / pullback.std()
            
            # 计算 alpha122：T 分数加权
            alpha = t_stat * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha122_{day}'] = alpha
        return self.data

    def alpha123(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that detects price gaps and volume anomalies using Z-scores.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算跳空：开盘与前一日收盘价的差值
            gap = close.diff(1)
            
            # 计算成交量的 Z 分数
            mean_volume = volume.rolling(window=day).mean()
            std_volume = volume.rolling(window=day).std()
            z_score_volume = (volume - mean_volume) / std_volume
            
            # 计算价格跳空的 Z 分数
            mean_gap = gap.rolling(window=day).mean()
            std_gap = gap.rolling(window=day).std()
            z_score_gap = (gap - mean_gap) / std_gap
            
            # 结合 Z 分数
            alpha = z_score_gap * z_score_volume * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha123_{day}'] = alpha
        return self.data

    def alpha124(self, days=list, anticipation_effect_factor=1.0):
        """
        Alpha indicator that combines price gap reversal and price momentum using T-scores.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            close = self.data['close']
            
            # 计算跳空：开盘与前一日收盘价的差值
            gap = close.diff(1)
            
            # 计算价格动量（收盘价差值）
            momentum = close.diff(day)
            
            # 计算 T 分数
            mean_gap = gap.rolling(window=day).mean()
            std_gap = gap.rolling(window=day).std()
            t_stat_gap = (gap - mean_gap) / std_gap
            
            mean_momentum = momentum.rolling(window=day).mean()
            std_momentum = momentum.rolling(window=day).std()
            t_stat_momentum = (momentum - mean_momentum) / std_momentum
            
            # 计算 alpha124：跳空反转与价格动量的加权
            alpha = (t_stat_gap * t_stat_momentum) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha124_{day}'] = alpha
        return self.data

    def alpha125(self, days=list, anticipation_effect_factor=1.0, rsi_threshold=70):
        """
        Alpha indicator that combines price gaps and RSI using Z-scores to detect overbought/oversold conditions.
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        - rsi_threshold (int): the threshold for detecting overbought conditions (default is 70)
        """
        for day in days:
            close = self.data['close']
            volume = self.data['volume']
            
            # 计算跳空：开盘与前一日收盘价的差值
            gap = close.diff(1)
            
            # 计算 RSI
            delta = close.diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=day).mean()
            avg_loss = loss.rolling(window=day).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # 计算 RSI 的 Z 分数
            mean_rsi = rsi.rolling(window=day).mean()
            std_rsi = rsi.rolling(window=day).std()
            z_score_rsi = (rsi - mean_rsi) / std_rsi
            
            # 计算跳空的 Z 分数
            mean_gap = gap.rolling(window=day).mean()
            std_gap = gap.rolling(window=day).std()
            z_score_gap = (gap - mean_gap) / std_gap
            
            # 结合 Z 分数与 RSI
            alpha = (z_score_gap * z_score_rsi) * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha125_{day}'] = alpha
        return self.data

    def alpha126(self, days=list, anticipation_effect_factor=3.0):
        """
        Alpha indicator that combines Fibonacci retracement levels and Golden Ratio (0.618).
        Parameter:
        - days (list): the time intervals for calculating indicators
        - anticipation_effect_factor (float): a factor to adjust the result (default is 1.0)
        """
        for day in days:
            close = self.data['close']
            
            # 计算当前周期的最高价和最低价
            high = close.rolling(window=day).max()
            low = close.rolling(window=day).min()
            
            # 计算费波那契回撤水平
            fibonacci_levels = {
                "level_0": low,
                "level_236": low + 0.236 * (high - low),
                "level_382": low + 0.382 * (high - low),
                "level_50": low + 0.5 * (high - low),
                "level_618": low + 0.618 * (high - low),
                "level_786": low + 0.786 * (high - low),
                "level_1": high
            }
            
            # 黄金分割比例：0.618
            golden_ratio = 0.618
            
            # 判断当前价格与黄金分割比例（0.618）和回撤水平的关系
            fib_retracement = (close - fibonacci_levels["level_618"]) / (fibonacci_levels["level_1"] - fibonacci_levels["level_0"])
            
            # 计算 alpha126：黄金分割与费波那契回撤结合
            alpha = fib_retracement * anticipation_effect_factor
            
            # 填充缺失值
            alpha = alpha.fillna(0)
            
            # 将结果添加到数据中
            self.data[f'alpha126_{day}'] = alpha
        return self.data

    def alpha127(self, days=list, threshold=2):
        """
        Calculate alpha127 using statistical deviation from historical price distribution.
        
        Parameters:
        - days (list): List of time intervals for the calculation.
        - threshold (float): Threshold for statistical significance, default is 3 (for 3-sigma rule).
        """
        new_columns = {}
        
        for day in days:
            # Calculate rolling mean and rolling standard deviation
            rolling_mean = self.data['close'].rolling(window=day).mean()
            rolling_std = self.data['close'].rolling(window=day).std()
            
            # Calculate the deviation of the current price from the rolling mean
            deviation = (self.data['close'] - rolling_mean) / (rolling_std + 1e-6)
            
            # Apply the threshold to capture significant deviations
            alpha = (deviation.abs() > threshold).astype(int)  # 1 if significant deviation, 0 otherwise
            
            new_columns[f'alpha127_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data



    def alpha128(self, days=list, volatility_scaler=2):
        """
        Calculate alpha128 using volatility and liquidity (trading volume) relationship.
        
        Parameters:
        - days (list): List of time intervals for the calculation.
        - volatility_scaler (float): Scaling factor for volatility (default is 1.0).
        """
        new_columns = {}
        
        for day in days:
            # Calculate rolling standard deviation (volatility)
            volatility = self.data['close'].pct_change().rolling(window=day).std()
            
            # Calculate volume-to-volatility ratio (liquidity factor)
            liquidity_factor = self.data['volume'] / (volatility + 1e-6)  # Avoid division by zero
            
            # Apply scaling factor for volatility
            alpha = liquidity_factor * volatility_scaler
            
            new_columns[f'alpha128_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data


    def alpha129(self, days=list):
        """
        通过统计指标（波动率、自相关性、变异系数等）设计 alpha129。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        import numpy as np

        new_columns = {}
        
        for day in days:
            # 计算波动率（标准差）
            volatility = self.data['close'].pct_change().rolling(window=day).std()

            # 计算变异系数（变动的标准差与均值的比值）
            mean_price = self.data['close'].rolling(window=day).mean()
            cv = volatility / mean_price  # 变异系数 = 波动率 / 均值

            # 计算自相关性（lag=1, 即与前一日的价格相关）
            autocorr = self.data['close'].pct_change().rolling(window=day).apply(lambda x: x.autocorr(lag=1))

            # 结合波动率、变异系数和自相关性进行加权求和生成 alpha129
            alpha = (volatility + cv + autocorr) / 3  # 平均法，或根据不同情况加权
            
            # 存储新生成的 alpha129
            new_columns[f'alpha129_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data





    def alpha130(self, days=list, volatility_scaler=0.2, conformity_factor=1.0):
        """
        Calculate alpha130 with multiple days using risk perception and herd behavior.

        Parameters:
        - days (list): List of time intervals for which the calculations are to be done.
        - volatility_scaler (float): A scaling factor for volatility (default is 0.2).
        - conformity_factor (float): Factor that adjusts herd behavior (default is 1.0).
        """
        # Create a dictionary to hold all the new columns to be added to the DataFrame
        new_columns = {}

        for day in days:
            # Calculate risk perception (avoiding division by zero or NaN)
            rolling_volatility = self.data['close'].pct_change().rolling(window=day).std()
            risk_perception = self.data['close'].pct_change(day) * (1 + volatility_scaler * rolling_volatility)
            new_columns[f'risk_perception_{day}'] = risk_perception

            # Calculate herd behavior
            avg_movement = self.data['close'].rolling(window=day).mean()
            herd_behavior = (self.data['close'] - avg_movement) * conformity_factor
            new_columns[f'herd_behavior_{day}'] = herd_behavior

            # Calculate alpha130
            alpha = (self.data['close'] - avg_movement) / (rolling_volatility + 1e-6)  # Add small constant to avoid division by zero
            new_columns[f'alpha130_{day}'] = alpha

        # Concatenate all new columns to the original DataFrame in one go
        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)

        return self.data


    def alpha131(self, days=list):
        """
        利用开盘价和收盘价之间的相对变化幅度来捕捉趋势或盘整市场。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算开盘价和收盘价的相对变化
            open_close_diff = (self.data['close'] - self.data['open']).rolling(window=day).mean()

            # 取绝对值来衡量波动的幅度
            alpha = open_close_diff.abs()

            new_columns[f'alpha131_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha132(self, days=list):
        """
        通过成交量变化与价格变化率之间的相关性来捕捉趋势或盘整市场。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算收盘价变化率
            price_change_rate = self.data['close'].pct_change()

            # 计算成交量的变化率
            volume_change_rate = self.data['volume'].pct_change()

            # 计算价格变化与成交量变化的滚动相关性
            correlation = price_change_rate.rolling(window=day).corr(volume_change_rate)

            new_columns[f'alpha132_{day}'] = correlation

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha133(self, days=list):
        """
        使用最高价和最低价之间的波动范围来捕捉市场的趋势或盘整特征。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算最高价和最低价之间的差距
            high_low_diff = (self.data['high'] - self.data['low']).rolling(window=day).mean()

            new_columns[f'alpha133_{day}'] = high_low_diff

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha134(self, days=list):
        """
        使用收盘价与前一天收盘价的波动率来判断市场是否处于趋势或盘整状态。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算收盘价的变化率（相对前一天）
            close_to_close_change = self.data['close'].pct_change().rolling(window=day).std()

            new_columns[f'alpha134_{day}'] = close_to_close_change

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha135(self, days=list):
        """
        通过计算收盘价与均线的偏离程度来判断市场的趋势性或盘整特征。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算均线
            moving_average = self.data['close'].rolling(window=day).mean()

            # 计算收盘价与均线的偏离程度
            deviation = (self.data['close'] - moving_average) / (moving_average + 1e-6)

            new_columns[f'alpha135_{day}'] = deviation

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha136(self, days=list):
        """
        成交量加权的波动率调整收益率。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}
        
        for day in days:
            # 计算收盘价的收益率
            returns = self.data['close'].pct_change()

            # 计算波动率（标准差）
            volatility = self.data['close'].pct_change().rolling(window=day).std()

            # 计算成交量的移动平均
            volume_ma = self.data['volume'].rolling(window=day).mean()

            # 结合收益率、波动率、和成交量的加权
            alpha = (returns / (volatility + 1e-6)) * volume_ma

            new_columns[f'alpha136_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha137(self, days=list):
        """
        利用非对称波动率来捕捉市场的趋势和风险。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算收益率
            returns = self.data['close'].pct_change()

            # 分别计算上涨和下跌的波动率
            up_volatility = returns[returns > 0].rolling(window=day).std()
            down_volatility = returns[returns < 0].rolling(window=day).std()

            # 计算成交量的滚动平均
            volume_ma = self.data['volume'].rolling(window=day).mean()

            # 非对称波动率因子，上涨波动率和下跌波动率相加后加权
            asym_vol = (up_volatility.fillna(0) - down_volatility.fillna(0)) * volume_ma

            new_columns[f'alpha137_{day}'] = asym_vol

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha138(self, short_window=10, mid_window=30, long_window=60, days=list):
        """
        使用多个移动均线的偏离度来评估市场趋势。
        
        参数:
        - short_window (int): 短期移动均线窗口大小。
        - mid_window (int): 中期移动均线窗口大小。
        - long_window (int): 长期移动均线窗口大小。
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算短期、中期和长期的移动均线
            short_ma = self.data['close'].rolling(window=short_window).mean()
            mid_ma = self.data['close'].rolling(window=mid_window).mean()
            long_ma = self.data['close'].rolling(window=long_window).mean()

            # 计算短期与中期，以及中期与长期均线之间的偏离程度
            short_mid_diff = (short_ma - mid_ma) / (mid_ma + 1e-6)
            mid_long_diff = (mid_ma - long_ma) / (long_ma + 1e-6)

            # 结合短期、中期、长期的偏离度
            alpha = (short_mid_diff + mid_long_diff) * day

            new_columns[f'alpha138_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha139(self, days=list):
        """
        使用高低价差与波动率结合来捕捉市场的波动特征。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算高低价差
            high_low_spread = self.data['high'] - self.data['low']

            # 计算波动率
            volatility = self.data['close'].pct_change().rolling(window=day).std()

            # 高低价差乘以波动率，捕捉市场在波动性增加时的变化特征
            alpha = high_low_spread * (1 + volatility)

            new_columns[f'alpha139_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha140(self, days=list):
        """
        使用价格的累积差异与成交量的交互效应来评估市场的趋势变化。
        
        参数:
        - days (list): 用于计算的时间间隔。
        """
        new_columns = {}

        for day in days:
            # 计算开盘价和收盘价的累积差异
            cumulative_diff = (self.data['close'] - self.data['open']).rolling(window=day).sum()

            # 计算成交量的移动平均
            volume_ma = self.data['volume'].rolling(window=day).mean()

            # 价格差异与成交量的交互效应
            alpha = cumulative_diff * volume_ma

            new_columns[f'alpha140_{day}'] = alpha

        self.data = pd.concat([self.data, pd.DataFrame(new_columns, index=self.data.index)], axis=1)
        return self.data

    def alpha141(self, days=list, return_weight=0.5):
        """
        Alpha141：结合波动率和动量的因子，使用滚动Z分数和收益率捕捉动量特征。
        
        参数:
        - days (list): 用于计算的时间间隔。
        - return_weight (float): 用于近期收益与波动率的权重比例。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            close = self.data['close']
            returns = close.pct_change()
            rolling_mean = close.rolling(window=day).mean()
            rolling_std = close.rolling(window=day).std()
            
            z_score = (close - rolling_mean) / rolling_std
            momentum = returns.rolling(window=day).apply(lambda x: sum(x), raw=True)

            alpha = (z_score * return_weight + momentum * (1 - return_weight))
            self.data[f'alpha141_{day}'] = alpha
        
        return self.data

    def alpha142(self, days=list, vol_weight=0.7):
        """
        Alpha142：基于波动性和均值回复特征的因子，使用成交量加权的标准差捕捉市场极端价格行为。
        
        参数:
        - days (list): 用于计算的时间间隔。
        - vol_weight (float): 波动性与成交量之间的加权比例。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            volume = self.data['volume']

            rolling_volatility = close.rolling(window=day).std()
            volume_scaled = volume.rolling(window=day).mean()

            alpha = (high - low) / rolling_volatility * vol_weight + volume_scaled * (1 - vol_weight)
            self.data[f'alpha142_{day}'] = alpha
        
        return self.data

    def alpha143(self, days=list, skew_weight=0.6):
        """
        Alpha143：通过偏态（Skewness）与滚动中位数的组合来捕捉价格分布的不对称性。
        
        参数:
        - days (list): 用于计算的时间间隔。
        - skew_weight (float): 偏度在最终 alpha 值中的权重比例。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            close = self.data['close']

            median_price = close.rolling(window=day).median()
            skewness = close.rolling(window=day).skew()

            alpha = (close - median_price) * skew_weight + skewness * (1 - skew_weight)
            self.data[f'alpha143_{day}'] = alpha
        
        return self.data

    def alpha144(self, days=list, kurt_weight=0.8):
        """
        Alpha144：使用峰度（Kurtosis）来检测市场行为的突然变化，高峰度表示价格的剧烈变化。
        
        参数:
        - days (list): 用于计算的时间间隔。
        - kurt_weight (float): 峰度在检测价格高峰条件下的权重。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            close = self.data['close']
            volume = self.data['volume']

            rolling_kurt = close.rolling(window=day).kurt()
            volume_mean = volume.rolling(window=day).mean()

            alpha = (rolling_kurt * kurt_weight) + volume_mean * (1 - kurt_weight)
            self.data[f'alpha144_{day}'] = alpha
        
        return self.data

    def alpha145(self, days=list, vix_relation_factor=0.3):
        """
        Alpha145：使用高低价格差和成交量的波动性关系，模拟与VIX类似的市场恐慌情绪。
        
        参数:
        - days (list): 用于计算的时间间隔。
        - vix_relation_factor (float): 波动率与成交量信号之间的平衡因子。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            high = self.data['high']
            low = self.data['low']
            volume = self.data['volume']

            range_volatility = (high - low) / (high + low + 0.001)
            rolling_vol = range_volatility.rolling(window=day).mean()
            volume_std = volume.rolling(window=day).std()

            alpha = rolling_vol * vix_relation_factor + volume_std * (1 - vix_relation_factor)
            self.data[f'alpha145_{day}'] = alpha

        return self.data
 
    def alpha146(self, days=list):
        """
        Alpha146：基于高低价与收盘价的差值，衡量短期波动性。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_range = self.data['high'] - self.data['low']
            close_diff = self.data['close'] - self.data['low']
            alpha = close_diff / (price_range + 1e-6)
            self.data[f'alpha146_{day}'] = alpha
        return self.data

    def alpha147(self, days=list):
        """
        Alpha147：利用开盘与收盘价的差异结合成交量，评估市场强度。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            open_close_diff = self.data['close'] - self.data['open']
            volume_mean = self.data['volume'].rolling(window=day).mean()
            alpha = open_close_diff * volume_mean
            self.data[f'alpha147_{day}'] = alpha
        return self.data

    def alpha148(self, days=list):
        """
        Alpha148：通过收盘价与均值价格的偏离度衡量市场的相对强弱。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            mean_price = (self.data['high'] + self.data['low']) / 2
            alpha = (self.data['close'] - mean_price) / mean_price
            self.data[f'alpha148_{day}'] = alpha
        return self.data

    def alpha149(self, days=list):
        """
        Alpha149：结合最高价、最低价与收盘价的加权移动平均，捕捉价格趋势。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            weighted_avg = (self.data['high'] + self.data['low'] + 2 * self.data['close']) / 4
            alpha = weighted_avg.rolling(window=day).mean()
            self.data[f'alpha149_{day}'] = alpha
        return self.data

    def alpha150(self, days=list):
        """
        Alpha150：利用成交量与价格波动的比率来衡量市场活跃度。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_volatility = self.data['high'] - self.data['low']
            volume_mean = self.data['volume'].rolling(window=day).mean()
            alpha = volume_mean / (price_volatility + 1e-6)
            self.data[f'alpha150_{day}'] = alpha
        return self.data

    def alpha151(self, days=list):
        """
        Alpha151：基于收盘价与开盘价之间的变化幅度，结合成交量变化。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_change = self.data['close'] - self.data['open']
            volume_std = self.data['volume'].rolling(window=day).std()
            alpha = price_change * volume_std
            self.data[f'alpha151_{day}'] = alpha
        return self.data

    def alpha152(self, days=list):
        """
        Alpha152：通过收盘价与开盘价的均值与价格波动幅度，评估市场的平稳程度。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            mean_price = (self.data['close'] + self.data['open']) / 2
            price_range = self.data['high'] - self.data['low']
            alpha = mean_price / (price_range + 1e-6)
            self.data[f'alpha152_{day}'] = alpha
        return self.data

    def alpha153(self, days=list):
        """
        Alpha153：计算短期内价格的变化速率，评估市场的动量特征。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_diff = self.data['close'].diff(day)
            alpha = price_diff / day
            self.data[f'alpha153_{day}'] = alpha
        return self.data

    def alpha154(self, days=list):
        """
        Alpha154：利用收盘价相对于高低价的相对位置，衡量价格的强弱。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_range = self.data['high'] - self.data['low']
            alpha = (self.data['close'] - self.data['low']) / (price_range + 1e-6)
            self.data[f'alpha154_{day}'] = alpha
        return self.data

    def alpha155(self, days=list):
        """
        Alpha155：基于成交量与价格波动的标准差，捕捉市场中的波动性特征。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_std = self.data['close'].rolling(window=day).std()
            volume_mean = self.data['volume'].rolling(window=day).mean()
            alpha = price_std * volume_mean
            self.data[f'alpha155_{day}'] = alpha
        return self.data

    def alpha156(self, days=list):
        """
        Alpha156：通过短周期内的收盘价变化幅度与成交量的相关性，衡量市场的波动性与交易活跃度。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_change = self.data['close'].diff()
            volume = self.data['volume']
            correlation = price_change.rolling(window=day).corr(volume)
            self.data[f'alpha156_{day}'] = correlation
        return self.data

    def alpha157(self, days=list):
        """
        Alpha157：利用开盘价和收盘价之间的差异与成交量标准差的组合来衡量市场动量。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            open_close_diff = self.data['close'] - self.data['open']
            volume_std = self.data['volume'].rolling(window=day).std()
            alpha = open_close_diff * volume_std
            self.data[f'alpha157_{day}'] = alpha
        return self.data

    def alpha158(self, days=list):
        """
        Alpha158：结合最高价、最低价与收盘价的标准差，捕捉市场的波动性。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            price_std = self.data[['high', 'low', 'close']].std(axis=1)
            rolling_std = price_std.rolling(window=day).mean()
            self.data[f'alpha158_{day}'] = rolling_std
        return self.data

    def alpha159(self, days=list):
        """
        Alpha159：使用成交量的变异系数（标准差/均值）来衡量市场交易的波动性。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            volume_mean = self.data['volume'].rolling(window=day).mean()
            volume_std = self.data['volume'].rolling(window=day).std()
            alpha = volume_std / (volume_mean + 1e-6)
            self.data[f'alpha159_{day}'] = alpha
        return self.data

    def alpha160(self, days=list):
        """
        Alpha160：结合收盘价的移动平均与成交量变化，捕捉市场趋势的强弱。
        
        参数:
        - days (list): 用于计算的时间间隔。
        
        返回:
        - df (pd.DataFrame): 返回包含新特征的数据。
        """
        for day in days:
            close_ma = self.data['close'].rolling(window=day).mean()
            volume_change = self.data['volume'].pct_change()
            alpha = close_ma * volume_change
            self.data[f'alpha160_{day}'] = alpha
        return self.data
       
    def add_all_alphas(self, days=[5, 10, 20, 60, 120, 240], custom_params=None):
        """
        Method to add all alpha features to the data at once.
        
        Parameter:
        - days (list): list of time intervals for each alpha (default: [5, 10, 20, 60, 120, 240])
        - custom_params (dict): a dictionary where keys are alpha method names and values are dictionaries of parameters 
          to override default parameters of the specific alpha. e.g., {"alpha12": {"adjustment_factor": 0.2}}
        
        Output:
        - self.data (pd.DataFrame): return the input df with all alpha series added
        """
        # List of alpha methods that use 'spread'
        spread_alphas = ['alpha46', 'alpha38','alpha30']  # Replace 'alphaXX' and 'alphaYY' with actual alpha method names that use 'spread'

        for method_name in dir(self):
            if method_name.startswith('alpha') and method_name[5:].isdigit():
                if method_name in spread_alphas:
                    continue  # Skip this method if it uses 'spread'

                method = getattr(self, method_name)
                if callable(method):
                    # If custom_params contains parameters for this method, use them
                    params = custom_params.get(method_name, {}) if custom_params else {}
                    # Call the method with the default 'days' list and overridden parameters if any
                    method(days=days, **params)

        return self.data
    
    

#Rolling alpha 