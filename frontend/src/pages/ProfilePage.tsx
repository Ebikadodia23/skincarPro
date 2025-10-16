import React from 'react';
import { useAuthStore } from '../store/authStore';

const ProfilePage: React.FC = () => {
  const { user } = useAuthStore();

  return (
    <div className="max-w-3xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Profile Settings</h1>
        <p className="text-gray-600">Manage your account information</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm p-6 border border-gray-200">
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Email</label>
            <input
              type="email"
              value={user?.email || ''}
              readOnly
              className="w-full px-4 py-3 border border-gray-300 rounded-lg bg-gray-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Username</label>
            <input
              type="text"
              value={user?.username || ''}
              readOnly
              className="w-full px-4 py-3 border border-gray-300 rounded-lg bg-gray-50"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfilePage;
