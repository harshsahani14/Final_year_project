import React, { useState } from 'react';
import { Shield, AlertTriangle, CheckCircle, User, Activity, Network, FileText, Calendar, Users, MessageSquare, Heart, Share2 } from 'lucide-react';

interface ProfileData {
  username: string;
  displayName: string;
  bio: string;
  profilePicture: string;
  followers: number;
  following: number;
  posts: number;
  accountAge: number; // in months
  verified: boolean;
  avgLikesPerPost: number;
  avgCommentsPerPost: number;
  postsPerWeek: number;
  profileCompleteness: number; // percentage
}

interface AnalysisResult {
  overallRisk: number;
  indicators: {
    profileCompleteness: { score: number; status: 'safe' | 'warning' | 'danger' };
    activityPatterns: { score: number; status: 'safe' | 'warning' | 'danger' };
    networkMetrics: { score: number; status: 'safe' | 'warning' | 'danger' };
    contentQuality: { score: number; status: 'safe' | 'warning' | 'danger' };
    accountAge: { score: number; status: 'safe' | 'warning' | 'danger' };
  };
}

function App() {
  const [profileData, setProfileData] = useState<ProfileData>({
    username: '',
    displayName: '',
    bio: '',
    profilePicture: '',
    followers: 0,
    following: 0,
    posts: 0,
    accountAge: 0,
    verified: false,
    avgLikesPerPost: 0,
    avgCommentsPerPost: 0,
    postsPerWeek: 0,
    profileCompleteness: 0
  });

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const analyzeProfile = () => {
    setIsAnalyzing(true);
    
    // Simulate analysis delay
    setTimeout(() => {
      const result = performAnalysis(profileData);
      setAnalysis(result);
      setIsAnalyzing(false);
    }, 2000);
  };

  const performAnalysis = (data: ProfileData): AnalysisResult => {
    // Profile Completeness Analysis
    let completenessScore = 0;
    if (data.username) completenessScore += 20;
    if (data.displayName) completenessScore += 20;
    if (data.bio) completenessScore += 25;
    if (data.profilePicture) completenessScore += 35;

    const profileCompleteness = {
      score: completenessScore,
      status: completenessScore >= 80 ? 'safe' : completenessScore >= 50 ? 'warning' : 'danger' as const
    };

    // Activity Patterns Analysis
    const followersToFollowing = data.followers > 0 ? data.following / data.followers : data.following;
    const likesToFollowers = data.followers > 0 ? data.avgLikesPerPost / data.followers : 0;
    
    let activityScore = 100;
    if (followersToFollowing > 2) activityScore -= 30; // Following too many people
    if (data.postsPerWeek > 20) activityScore -= 25; // Posting too frequently
    if (data.postsPerWeek < 1 && data.accountAge > 6) activityScore -= 20; // Too inactive
    if (likesToFollowers > 0.1) activityScore -= 15; // Unusual engagement ratio

    const activityPatterns = {
      score: Math.max(0, activityScore),
      status: activityScore >= 70 ? 'safe' : activityScore >= 40 ? 'warning' : 'danger' as const
    };

    // Network Metrics Analysis
    const followerToPostRatio = data.posts > 0 ? data.followers / data.posts : 0;
    let networkScore = 100;
    
    if (data.followers > 1000 && data.posts < 10) networkScore -= 40; // Too many followers for content
    if (data.followers < 50 && data.accountAge > 12) networkScore -= 20; // Low followers for old account
    if (followerToPostRatio > 100) networkScore -= 30; // Suspicious follower growth

    const networkMetrics = {
      score: Math.max(0, networkScore),
      status: networkScore >= 70 ? 'safe' : networkScore >= 40 ? 'warning' : 'danger' as const
    };

    // Content Quality Analysis
    const engagementRate = data.followers > 0 ? (data.avgLikesPerPost + data.avgCommentsPerPost) / data.followers : 0;
    let contentScore = 100;
    
    if (engagementRate < 0.01 && data.followers > 500) contentScore -= 30; // Low engagement
    if (engagementRate > 0.15) contentScore -= 20; // Suspiciously high engagement
    if (data.bio.length < 10) contentScore -= 25; // Poor bio quality

    const contentQuality = {
      score: Math.max(0, contentScore),
      status: contentScore >= 70 ? 'safe' : contentScore >= 40 ? 'warning' : 'danger' as const
    };

    // Account Age Analysis
    let ageScore = 100;
    if (data.accountAge < 3) ageScore = 30; // Very new account
    else if (data.accountAge < 6) ageScore = 60; // Relatively new
    else if (data.accountAge < 12) ageScore = 80; // Moderate age
    
    if (data.verified) ageScore = Math.min(100, ageScore + 20); // Verified bonus

    const accountAge = {
      score: ageScore,
      status: ageScore >= 70 ? 'safe' : ageScore >= 40 ? 'warning' : 'danger' as const
    };

    // Calculate overall risk
    const scores = [
      profileCompleteness.score,
      activityPatterns.score,
      networkMetrics.score,
      contentQuality.score,
      accountAge.score
    ];
    
    const overallRisk = 100 - (scores.reduce((sum, score) => sum + score, 0) / scores.length);

    return {
      overallRisk,
      indicators: {
        profileCompleteness,
        activityPatterns,
        networkMetrics,
        contentQuality,
        accountAge
      }
    };
  };

  const getRiskLevel = (risk: number) => {
    if (risk >= 70) return { level: 'High Risk', color: 'text-red-500', bg: 'bg-red-500' };
    if (risk >= 40) return { level: 'Medium Risk', color: 'text-yellow-500', bg: 'bg-yellow-500' };
    return { level: 'Low Risk', color: 'text-green-500', bg: 'bg-green-500' };
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'safe': return 'text-green-500';
      case 'warning': return 'text-yellow-500';
      case 'danger': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'safe': return <CheckCircle className="w-5 h-5" />;
      case 'warning': return <AlertTriangle className="w-5 h-5" />;
      case 'danger': return <AlertTriangle className="w-5 h-5" />;
      default: return <AlertTriangle className="w-5 h-5" />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center">
            <Shield className="w-8 h-8 text-blue-500 mr-3" />
            <h1 className="text-2xl font-bold">Social Account Authenticity Analyzer</h1>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Form */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-6 flex items-center">
              <User className="w-5 h-5 mr-2 text-blue-500" />
              Profile Data Input
            </h2>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Username</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.username}
                    onChange={(e) => setProfileData({...profileData, username: e.target.value})}
                    placeholder="@username"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Display Name</label>
                  <input
                    type="text"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.displayName}
                    onChange={(e) => setProfileData({...profileData, displayName: e.target.value})}
                    placeholder="Display Name"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Bio</label>
                <textarea
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={3}
                  value={profileData.bio}
                  onChange={(e) => setProfileData({...profileData, bio: e.target.value})}
                  placeholder="Profile bio..."
                />
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Followers</label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.followers}
                    onChange={(e) => setProfileData({...profileData, followers: parseInt(e.target.value) || 0})}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Following</label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.following}
                    onChange={(e) => setProfileData({...profileData, following: parseInt(e.target.value) || 0})}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Posts</label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.posts}
                    onChange={(e) => setProfileData({...profileData, posts: parseInt(e.target.value) || 0})}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Account Age (months)</label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.accountAge}
                    onChange={(e) => setProfileData({...profileData, accountAge: parseInt(e.target.value) || 0})}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Posts/Week</label>
                  <input
                    type="number"
                    step="0.1"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.postsPerWeek}
                    onChange={(e) => setProfileData({...profileData, postsPerWeek: parseFloat(e.target.value) || 0})}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Avg Likes/Post</label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.avgLikesPerPost}
                    onChange={(e) => setProfileData({...profileData, avgLikesPerPost: parseInt(e.target.value) || 0})}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Avg Comments/Post</label>
                  <input
                    type="number"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    value={profileData.avgCommentsPerPost}
                    onChange={(e) => setProfileData({...profileData, avgCommentsPerPost: parseInt(e.target.value) || 0})}
                  />
                </div>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="verified"
                  className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                  checked={profileData.verified}
                  onChange={(e) => setProfileData({...profileData, verified: e.target.checked})}
                />
                <label htmlFor="verified" className="ml-2 text-sm font-medium text-gray-300">
                  Verified Account
                </label>
              </div>

              <button
                onClick={analyzeProfile}
                disabled={isAnalyzing}
                className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:opacity-50 text-white font-medium py-3 px-4 rounded-md transition duration-200 flex items-center justify-center"
              >
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Shield className="w-5 h-5 mr-2" />
                    Analyze Account
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Analysis Results */}
          <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-6 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-blue-500" />
              Analysis Results
            </h2>

            {!analysis ? (
              <div className="text-center text-gray-400 py-12">
                <Shield className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p>Enter profile data and click "Analyze Account" to see results</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Overall Risk Score */}
                <div className="bg-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-medium">Overall Risk Assessment</h3>
                    <span className={`text-2xl font-bold ${getRiskLevel(analysis.overallRisk).color}`}>
                      {analysis.overallRisk.toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-600 rounded-full h-3 mb-2">
                    <div 
                      className={`h-3 rounded-full ${getRiskLevel(analysis.overallRisk).bg}`}
                      style={{ width: `${analysis.overallRisk}%` }}
                    ></div>
                  </div>
                  <p className={`text-sm font-medium ${getRiskLevel(analysis.overallRisk).color}`}>
                    {getRiskLevel(analysis.overallRisk).level}
                  </p>
                </div>

                {/* Detailed Indicators */}
                <div className="space-y-4">
                  <h3 className="text-lg font-medium">Detection Indicators</h3>
                  
                  {Object.entries(analysis.indicators).map(([key, indicator]) => {
                    const labels = {
                      profileCompleteness: 'Profile Completeness',
                      activityPatterns: 'Activity Patterns',
                      networkMetrics: 'Network Metrics',
                      contentQuality: 'Content Quality',
                      accountAge: 'Account Age'
                    };
                    
                    const icons = {
                      profileCompleteness: <User className="w-4 h-4" />,
                      activityPatterns: <Activity className="w-4 h-4" />,
                      networkMetrics: <Network className="w-4 h-4" />,
                      contentQuality: <FileText className="w-4 h-4" />,
                      accountAge: <Calendar className="w-4 h-4" />
                    };

                    return (
                      <div key={key} className="bg-gray-700 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center">
                            <span className={`mr-2 ${getStatusColor(indicator.status)}`}>
                              {icons[key as keyof typeof icons]}
                            </span>
                            <span className="font-medium">{labels[key as keyof typeof labels]}</span>
                          </div>
                          <div className="flex items-center">
                            <span className={`mr-2 ${getStatusColor(indicator.status)}`}>
                              {getStatusIcon(indicator.status)}
                            </span>
                            <span className={`font-bold ${getStatusColor(indicator.status)}`}>
                              {indicator.score}%
                            </span>
                          </div>
                        </div>
                        <div className="w-full bg-gray-600 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full ${
                              indicator.status === 'safe' ? 'bg-green-500' :
                              indicator.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                            style={{ width: `${indicator.score}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Recommendations */}
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-lg font-medium mb-3 flex items-center">
                    <AlertTriangle className="w-5 h-5 mr-2 text-yellow-500" />
                    Recommendations
                  </h3>
                  <div className="space-y-2 text-sm text-gray-300">
                    {analysis.overallRisk >= 70 && (
                      <p>• High risk detected - recommend manual review and additional verification</p>
                    )}
                    {analysis.indicators.profileCompleteness.status !== 'safe' && (
                      <p>• Incomplete profile information may indicate automated account creation</p>
                    )}
                    {analysis.indicators.activityPatterns.status !== 'safe' && (
                      <p>• Unusual activity patterns detected - review posting frequency and engagement</p>
                    )}
                    {analysis.indicators.networkMetrics.status !== 'safe' && (
                      <p>• Suspicious follower-to-content ratio - potential follower manipulation</p>
                    )}
                    {analysis.indicators.accountAge.status !== 'safe' && (
                      <p>• New account - monitor activity over time before making final determination</p>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;