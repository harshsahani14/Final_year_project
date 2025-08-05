import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from textblob import TextBlob
from typing import Dict, List, Any
import logging

class FeatureExtractor:
    """Extract features from user data for fake account detection"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_all_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract all features from user data"""
        try:
            features = {}
            
            # Profile features
            profile_features = self.extract_profile_features(user_data)
            features.update(profile_features)
            
            # Behavioral features
            behavioral_features = self.extract_behavioral_features(user_data)
            features.update(behavioral_features)
            
            # Content features
            content_features = self.extract_content_features(user_data)
            features.update(content_features)
            
            # Network features
            network_features = self.extract_network_features(user_data)
            features.update(network_features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            return self._get_default_features()
    
    def extract_profile_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract profile-based features"""
        features = {}
        
        # Account age in days
        creation_date = pd.to_datetime(user_data.get('creation_date', datetime.now()))
        account_age = (datetime.now() - creation_date).days
        features['account_age_days'] = float(account_age)
        
        # Follower/Following ratio
        followers = user_data.get('follower_count', 0)
        following = user_data.get('following_count', 1)  # Avoid division by zero
        features['follower_following_ratio'] = float(followers / max(following, 1))
        
        # Profile completeness score
        completeness_score = self._calculate_profile_completeness(user_data)
        features['profile_completeness'] = completeness_score
        
        # Username analysis
        username_score = self._analyze_username(user_data.get('username', ''))
        features['username_suspicion_score'] = username_score
        
        # Profile picture analysis
        has_profile_pic = 1.0 if user_data.get('profile_picture_url') else 0.0
        features['has_profile_picture'] = has_profile_pic
        
        # Bio analysis
        bio_score = self._analyze_bio(user_data.get('bio', ''))
        features['bio_quality_score'] = bio_score
        
        # Verification status
        is_verified = 1.0 if user_data.get('is_verified', False) else 0.0
        features['is_verified'] = is_verified
        
        return features
    
    def extract_behavioral_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract behavioral pattern features"""
        features = {}
        posts = user_data.get('posts', [])
        
        # Posting frequency (posts per day)
        if posts and len(posts) > 0:
            post_dates = [pd.to_datetime(post.get('created_at')) for post in posts]
            date_range = (max(post_dates) - min(post_dates)).days
            posting_frequency = len(posts) / max(date_range, 1)
        else:
            posting_frequency = 0.0
        
        features['posting_frequency'] = posting_frequency
        
        # Activity pattern analysis
        activity_pattern_score = self._analyze_activity_pattern(posts)
        features['activity_pattern_score'] = activity_pattern_score
        
        # Engagement rate
        engagement_rate = self._calculate_engagement_rate(posts)
        features['engagement_rate'] = engagement_rate
        
        # Time pattern regularity
        time_regularity = self._analyze_time_patterns(posts)
        features['time_pattern_regularity'] = time_regularity
        
        # Burst posting detection
        burst_score = self._detect_burst_posting(posts)
        features['burst_posting_score'] = burst_score
        
        return features
    
    def extract_content_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract content-based features"""
        features = {}
        posts = user_data.get('posts', [])
        
        if not posts:
            return {
                'avg_post_length': 0.0,
                'duplicate_content_ratio': 0.0,
                'spam_content_score': 0.0,
                'sentiment_variance': 0.0,
                'hashtag_abuse_score': 0.0
            }
        
        # Average post length
        post_lengths = [len(post.get('content', '')) for post in posts]
        features['avg_post_length'] = float(np.mean(post_lengths)) if post_lengths else 0.0
        
        # Duplicate content ratio
        duplicate_ratio = self._calculate_duplicate_content_ratio(posts)
        features['duplicate_content_ratio'] = duplicate_ratio
        
        # Spam content detection
        spam_score = self._detect_spam_content(posts)
        features['spam_content_score'] = spam_score
        
        # Sentiment analysis variance
        sentiment_variance = self._analyze_sentiment_variance(posts)
        features['sentiment_variance'] = sentiment_variance
        
        # Hashtag abuse detection
        hashtag_score = self._analyze_hashtag_usage(posts)
        features['hashtag_abuse_score'] = hashtag_score
        
        return features
    
    def extract_network_features(self, user_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract network-based features"""
        features = {}
        
        # Follower quality score (simplified)
        follower_quality = self._estimate_follower_quality(user_data)
        features['follower_quality_score'] = follower_quality
        
        # Following pattern analysis
        following_pattern = self._analyze_following_pattern(user_data)
        features['following_pattern_score'] = following_pattern
        
        # Mutual connections ratio
        mutual_ratio = self._calculate_mutual_connections_ratio(user_data)
        features['mutual_connections_ratio'] = mutual_ratio
        
        return features
    
    def _calculate_profile_completeness(self, user_data: Dict[str, Any]) -> float:
        """Calculate profile completeness score"""
        completeness_factors = [
            bool(user_data.get('username')),
            bool(user_data.get('bio')),
            bool(user_data.get('profile_picture_url')),
            bool(user_data.get('location')),
            bool(user_data.get('website_url')),
            user_data.get('follower_count', 0) > 0,
            user_data.get('following_count', 0) > 0
        ]
        
        return float(sum(completeness_factors) / len(completeness_factors))
    
    def _analyze_username(self, username: str) -> float:
        """Analyze username for suspicious patterns"""
        if not username:
            return 1.0  # Missing username is suspicious
        
        suspicion_score = 0.0
        
        # Check for random character patterns
        if re.search(r'\d{4,}', username):  # 4+ consecutive digits
            suspicion_score += 0.3
        
        # Check for excessive underscores or numbers
        if username.count('_') > 2 or len(re.findall(r'\d', username)) > len(username) * 0.5:
            suspicion_score += 0.2
        
        # Check for common bot patterns
        bot_patterns = ['bot', 'fake', 'spam', 'auto']
        if any(pattern in username.lower() for pattern in bot_patterns):
            suspicion_score += 0.5
        
        return min(suspicion_score, 1.0)
    
    def _analyze_bio(self, bio: str) -> float:
        """Analyze bio quality"""
        if not bio:
            return 0.3  # No bio is somewhat suspicious
        
        quality_score = 0.0
        
        # Length check
        if 10 <= len(bio) <= 200:
            quality_score += 0.3
        
        # Language quality
        try:
            blob = TextBlob(bio)
            if blob.sentiment.polarity != 0:  # Has sentiment
                quality_score += 0.2
        except:
            pass
        
        # Check for spam patterns
        spam_patterns = ['follow me', 'check out', 'click here', 'buy now']
        if not any(pattern in bio.lower() for pattern in spam_patterns):
            quality_score += 0.3
        
        # Personal information presence
        if any(word in bio.lower() for word in ['love', 'family', 'work', 'student', 'from']):
            quality_score += 0.2
        
        return quality_score
    
    def _analyze_activity_pattern(self, posts: List[Dict]) -> float:
        """Analyze posting activity patterns"""
        if len(posts) < 5:
            return 0.5  # Not enough data
        
        # Convert timestamps to hours
        post_hours = []
        for post in posts:
            try:
                post_time = pd.to_datetime(post.get('created_at'))
                post_hours.append(post_time.hour)
            except:
                continue
        
        if not post_hours:
            return 0.5
        
        # Calculate hour distribution
        hour_counts = np.bincount(post_hours, minlength=24)
        hour_distribution = hour_counts / len(post_hours)
        
        # Human-like activity should have some variation but not be completely random
        entropy = -np.sum(hour_distribution * np.log(hour_distribution + 1e-10))
        normalized_entropy = entropy / np.log(24)  # Normalize to 0-1
        
        # Score closer to 0.5-0.8 is more human-like
        if 0.3 <= normalized_entropy <= 0.8:
            return 1.0 - abs(normalized_entropy - 0.55) * 2
        else:
            return 0.3  # Too random or too regular
    
    def _calculate_engagement_rate(self, posts: List[Dict]) -> float:
        """Calculate average engagement rate"""
        if not posts:
            return 0.0
        
        engagement_rates = []
        for post in posts:
            likes = post.get('likes_count', 0)
            shares = post.get('shares_count', 0)
            comments = post.get('comments_count', 0)
            
            total_engagement = likes + shares + comments
            engagement_rates.append(total_engagement)
        
        return float(np.mean(engagement_rates)) if engagement_rates else 0.0
    
    def _analyze_time_patterns(self, posts: List[Dict]) -> float:
        """Analyze time pattern regularity"""
        if len(posts) < 3:
            return 0.5
        
        try:
            post_times = [pd.to_datetime(post.get('created_at')) for post in posts]
            post_times.sort()
            
            # Calculate intervals between posts
            intervals = [(post_times[i+1] - post_times[i]).total_seconds() 
                        for i in range(len(post_times)-1)]
            
            if not intervals:
                return 0.5
            
            # Calculate coefficient of variation
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if mean_interval == 0:
                return 0.0
            
            cv = std_interval / mean_interval
            
            # Human posting should have some irregularity
            # CV too low (too regular) or too high (too random) is suspicious
            if 0.5 <= cv <= 2.0:
                return 1.0
            else:
                return max(0.0, 1.0 - abs(cv - 1.25) / 2.0)
                
        except Exception:
            return 0.5
    
    def _detect_burst_posting(self, posts: List[Dict]) -> float:
        """Detect burst posting patterns"""
        if len(posts) < 5:
            return 0.0
        
        try:
            post_times = [pd.to_datetime(post.get('created_at')) for post in posts]
            post_times.sort()
            
            # Count posts in 1-hour windows
            burst_score = 0.0
            for i, post_time in enumerate(post_times):
                # Count posts within 1 hour of this post
                hour_window = post_time + timedelta(hours=1)
                posts_in_hour = sum(1 for t in post_times[i:] if t <= hour_window)
                
                # More than 10 posts in an hour is suspicious
                if posts_in_hour > 10:
                    burst_score += 0.1
                elif posts_in_hour > 5:
                    burst_score += 0.05
            
            return min(burst_score, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_duplicate_content_ratio(self, posts: List[Dict]) -> float:
        """Calculate ratio of duplicate content"""
        if len(posts) < 2:
            return 0.0
        
        contents = [post.get('content', '').strip().lower() for post in posts]
        contents = [c for c in contents if c]  # Remove empty content
        
        if len(contents) < 2:
            return 0.0
        
        unique_contents = set(contents)
        duplicate_ratio = 1.0 - (len(unique_contents) / len(contents))
        
        return duplicate_ratio
    
    def _detect_spam_content(self, posts: List[Dict]) -> float:
        """Detect spam content patterns"""
        if not posts:
            return 0.0
        
        spam_indicators = [
            'buy now', 'click here', 'free money', 'make money fast',
            'work from home', 'lose weight fast', 'miracle cure',
            'limited time', 'act now', 'call now', 'visit our website'
        ]
        
        spam_score = 0.0
        for post in posts:
            content = post.get('content', '').lower()
            spam_count = sum(1 for indicator in spam_indicators if indicator in content)
            if spam_count > 0:
                spam_score += spam_count / len(spam_indicators)
        
        return min(spam_score / len(posts), 1.0) if posts else 0.0
    
    def _analyze_sentiment_variance(self, posts: List[Dict]) -> float:
        """Analyze sentiment variance in posts"""
        if not posts:
            return 0.0
        
        sentiments = []
        for post in posts:
            content = post.get('content', '')
            if content:
                try:
                    blob = TextBlob(content)
                    sentiments.append(blob.sentiment.polarity)
                except:
                    continue
        
        if len(sentiments) < 2:
            return 0.0
        
        # Calculate variance in sentiment
        sentiment_variance = np.var(sentiments)
        
        # Normalize variance (typical range is 0-1)
        normalized_variance = min(sentiment_variance * 2, 1.0)
        
        return normalized_variance
    
    def _analyze_hashtag_usage(self, posts: List[Dict]) -> float:
        """Analyze hashtag usage patterns"""
        if not posts:
            return 0.0
        
        hashtag_counts = []
        for post in posts:
            hashtags = post.get('hashtags', [])
            hashtag_counts.append(len(hashtags))
        
        if not hashtag_counts:
            return 0.0
        
        avg_hashtags = np.mean(hashtag_counts)
        max_hashtags = max(hashtag_counts)
        
        # Excessive hashtag usage is suspicious
        abuse_score = 0.0
        if avg_hashtags > 10:
            abuse_score += 0.3
        if max_hashtags > 20:
            abuse_score += 0.4
        if avg_hashtags > 15:
            abuse_score += 0.3
        
        return min(abuse_score, 1.0)
    
    def _estimate_follower_quality(self, user_data: Dict[str, Any]) -> float:
        """Estimate follower quality (simplified)"""
        followers = user_data.get('follower_count', 0)
        following = user_data.get('following_count', 0)
        posts = len(user_data.get('posts', []))
        
        if followers == 0:
            return 0.0
        
        # Simple heuristics for follower quality
        quality_score = 0.5  # Base score
        
        # Follower to post ratio
        if posts > 0:
            follower_post_ratio = followers / posts
            if 1 <= follower_post_ratio <= 100:  # Reasonable range
                quality_score += 0.2
        
        # Following to follower ratio
        if followers > 0:
            following_follower_ratio = following / followers
            if 0.1 <= following_follower_ratio <= 2.0:  # Reasonable range
                quality_score += 0.2
        
        # Account age factor
        creation_date = pd.to_datetime(user_data.get('creation_date', datetime.now()))
        account_age = (datetime.now() - creation_date).days
        if account_age > 30:  # Older accounts tend to have better followers
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _analyze_following_pattern(self, user_data: Dict[str, Any]) -> float:
        """Analyze following pattern"""
        following = user_data.get('following_count', 0)
        followers = user_data.get('follower_count', 0)
        
        # Suspicious patterns
        suspicion_score = 0.0
        
        # Following too many accounts
        if following > 5000:
            suspicion_score += 0.3
        elif following > 2000:
            suspicion_score += 0.1
        
        # Following much more than followers (except for new accounts)
        if followers > 0 and following / followers > 10:
            suspicion_score += 0.4
        
        # Following exactly round numbers (bot behavior)
        if following > 0 and following % 100 == 0 and following > 500:
            suspicion_score += 0.2
        
        return min(suspicion_score, 1.0)
    
    def _calculate_mutual_connections_ratio(self, user_data: Dict[str, Any]) -> float:
        """Calculate mutual connections ratio (simplified)"""
        # This would require network data which we don't have in this simplified version
        # Return a default value based on follower/following ratio
        followers = user_data.get('follower_count', 0)
        following = user_data.get('following_count', 0)
        
        if followers == 0 or following == 0:
            return 0.0
        
        # Estimate mutual connections based on follower/following balance
        ratio = min(followers, following) / max(followers, following)
        return ratio
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values in case of error"""
        return {
            'account_age_days': 0.0,
            'follower_following_ratio': 0.0,
            'profile_completeness': 0.0,
            'username_suspicion_score': 1.0,
            'has_profile_picture': 0.0,
            'bio_quality_score': 0.0,
            'is_verified': 0.0,
            'posting_frequency': 0.0,
            'activity_pattern_score': 0.0,
            'engagement_rate': 0.0,
            'time_pattern_regularity': 0.0,
            'burst_posting_score': 0.0,
            'avg_post_length': 0.0,
            'duplicate_content_ratio': 0.0,
            'spam_content_score': 0.0,
            'sentiment_variance': 0.0,
            'hashtag_abuse_score': 0.0,
            'follower_quality_score': 0.0,
            'following_pattern_score': 0.0,
            'mutual_connections_ratio': 0.0
        }
