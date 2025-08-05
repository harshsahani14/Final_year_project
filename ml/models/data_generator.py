import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import json

class SyntheticDataGenerator:
    """Generate synthetic social media data for training"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Common patterns for fake accounts
        self.fake_username_patterns = [
            'user{}', 'account{}', 'profile{}', 'bot{}', 
            'fake{}', 'spam{}', 'auto{}', 'test{}'
        ]
        
        self.real_username_patterns = [
            'john_doe', 'sarah_smith', 'mike_jones', 'lisa_brown',
            'alex_wilson', 'emma_davis', 'chris_taylor', 'anna_white'
        ]
        
        self.spam_content = [
            "Buy now! Limited time offer!",
            "Click here for amazing deals!",
            "Make money fast from home!",
            "Lose weight quickly with this miracle cure!",
            "Follow me for more content like this!",
            "Check out my website for exclusive offers!"
        ]
        
        self.real_content = [
            "Had a great day at the beach with family!",
            "Just finished reading an amazing book.",
            "Excited about the new project at work.",
            "Beautiful sunset today, feeling grateful.",
            "Weekend plans: hiking and relaxing.",
            "Trying out a new recipe for dinner tonight."
        ]
    
    def generate_dataset(self, n_samples: int = 1000, fake_ratio: float = 0.3) -> pd.DataFrame:
        """Generate synthetic dataset"""
        n_fake = int(n_samples * fake_ratio)
        n_real = n_samples - n_fake
        
        # Generate fake accounts
        fake_accounts = [self._generate_fake_account(i) for i in range(n_fake)]
        
        # Generate real accounts
        real_accounts = [self._generate_real_account(i) for i in range(n_real)]
        
        # Combine and shuffle
        all_accounts = fake_accounts + real_accounts
        random.shuffle(all_accounts)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_accounts)
        
        return df
    
    def _generate_fake_account(self, account_id: int) -> Dict[str, Any]:
        """Generate a fake account profile"""
        # Basic profile
        username = random.choice(self.fake_username_patterns).format(
            random.randint(1000, 9999)
        )
        
        # Fake accounts often have suspicious patterns
        creation_date = datetime.now() - timedelta(days=random.randint(1, 30))  # Recent creation
        
        # Suspicious follower patterns
        if random.random() < 0.3:  # 30% have very high following
            following_count = random.randint(2000, 10000)
            follower_count = random.randint(10, 100)
        else:
            following_count = random.randint(500, 2000)
            follower_count = random.randint(5, 200)
        
        # Generate posts (often spam or duplicate)
        posts = self._generate_fake_posts(random.randint(5, 50))
        
        return {
            'user_id': f'fake_{account_id}',
            'username': username,
            'email': f'{username}@example.com',
            'creation_date': creation_date.isoformat(),
            'follower_count': follower_count,
            'following_count': following_count,
            'is_verified': False,  # Fake accounts rarely verified
            'bio': self._generate_fake_bio(),
            'profile_picture_url': '' if random.random() < 0.4 else 'http://example.com/pic.jpg',
            'location': '',
            'website_url': '',
            'posts': posts,
            'is_fake': 1  # Label
        }
    
    def _generate_real_account(self, account_id: int) -> Dict[str, Any]:
        """Generate a real account profile"""
        # Basic profile
        username = random.choice(self.real_username_patterns) + str(random.randint(1, 999))
        
        # Real accounts have more natural creation dates
        creation_date = datetime.now() - timedelta(days=random.randint(30, 1000))
        
        # More balanced follower patterns
        follower_count = random.randint(50, 2000)
        following_count = random.randint(20, min(follower_count * 2, 1000))
        
        # Generate posts (more natural content)
        posts = self._generate_real_posts(random.randint(10, 200))
        
        return {
            'user_id': f'real_{account_id}',
            'username': username,
            'email': f'{username}@example.com',
            'creation_date': creation_date.isoformat(),
            'follower_count': follower_count,
            'following_count': following_count,
            'is_verified': random.random() < 0.1,  # 10% chance of verification
            'bio': self._generate_real_bio(),
            'profile_picture_url': 'http://example.com/pic.jpg',
            'location': random.choice(['New York', 'California', 'Texas', 'Florida', '']),
            'website_url': 'http://example.com' if random.random() < 0.3 else '',
            'posts': posts,
            'is_fake': 0  # Label
        }
    
    def _generate_fake_posts(self, n_posts: int) -> List[Dict[str, Any]]:
        """Generate fake posts with spam patterns"""
        posts = []
        base_time = datetime.now() - timedelta(days=30)
        
        for i in range(n_posts):
            # Fake accounts often post in bursts
            if random.random() < 0.3:  # 30% chance of burst posting
                post_time = base_time + timedelta(minutes=random.randint(0, 60))
            else:
                post_time = base_time + timedelta(hours=random.randint(0, 720))
            
            # Often duplicate or spam content
            if random.random() < 0.4:  # 40% spam content
                content = random.choice(self.spam_content)
            elif random.random() < 0.3:  # 30% duplicate content
                content = posts[0]['content'] if posts else "Duplicate content"
            else:
                content = random.choice(self.real_content)
            
            # Fake engagement patterns
            likes = random.randint(0, 10)  # Low engagement
            shares = random.randint(0, 2)
            comments = random.randint(0, 3)
            
            # Excessive hashtags
            hashtags = []
            if random.random() < 0.5:  # 50% chance of hashtags
                n_hashtags = random.randint(5, 20)  # Excessive hashtags
                hashtags = [f'#tag{j}' for j in range(n_hashtags)]
            
            posts.append({
                'post_id': f'post_{i}',
                'content': content,
                'created_at': post_time.isoformat(),
                'likes_count': likes,
                'shares_count': shares,
                'comments_count': comments,
                'hashtags': hashtags
            })
        
        return posts
    
    def _generate_real_posts(self, n_posts: int) -> List[Dict[str, Any]]:
        """Generate real posts with natural patterns"""
        posts = []
        base_time = datetime.now() - timedelta(days=365)
        
        for i in range(n_posts):
            # More natural posting times
            post_time = base_time + timedelta(
                days=random.randint(0, 365),
                hours=random.randint(6, 23),  # Avoid late night posting
                minutes=random.randint(0, 59)
            )
            
            # Natural content
            content = random.choice(self.real_content)
            
            # Natural engagement patterns
            likes = random.randint(5, 100)
            shares = random.randint(0, 20)
            comments = random.randint(0, 15)
            
            # Reasonable hashtags
            hashtags = []
            if random.random() < 0.3:  # 30% chance of hashtags
                n_hashtags = random.randint(1, 5)  # Reasonable number
                hashtags = [f'#tag{j}' for j in range(n_hashtags)]
            
            posts.append({
                'post_id': f'post_{i}',
                'content': content,
                'created_at': post_time.isoformat(),
                'likes_count': likes,
                'shares_count': shares,
                'comments_count': comments,
                'hashtags': hashtags
            })
        
        return posts
    
    def _generate_fake_bio(self) -> str:
        """Generate fake bio"""
        fake_bios = [
            "",  # Empty bio
            "Follow me for more!",
            "Check out my website!",
            "Buy my products now!",
            "Make money fast!",
            "Click the link in bio!"
        ]
        return random.choice(fake_bios)
    
    def _generate_real_bio(self) -> str:
        """Generate real bio"""
        real_bios = [
            "Love traveling and photography",
            "Software engineer from California",
            "Dog lover and coffee enthusiast",
            "Fitness enthusiast and healthy living",
            "Artist and creative soul",
            "Family first, work second",
            "Bookworm and movie buff",
            "Outdoor adventures and hiking"
        ]
        return random.choice(real_bios)
    
    def save_dataset(self, df: pd.DataFrame, filename: str) -> None:
        """Save dataset to file"""
        df.to_json(filename, orient='records', indent=2)
        print(f"Dataset saved to {filename}")
    
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load dataset from file"""
        return pd.read_json(filename)
