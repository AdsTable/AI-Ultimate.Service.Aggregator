-- Database migration script to fix missing columns and improve schema
-- Run this to resolve: no such column: extraction_method

-- Add missing columns to service_plans table
ALTER TABLE service_plans ADD COLUMN extraction_method TEXT DEFAULT 'unknown';
ALTER TABLE service_plans ADD COLUMN data_quality_score REAL DEFAULT 0.0;
ALTER TABLE service_plans ADD COLUMN last_verified TIMESTAMP;

-- Add missing columns to providers table  
ALTER TABLE providers ADD COLUMN extraction_method TEXT DEFAULT 'unknown';
ALTER TABLE providers ADD COLUMN is_active BOOLEAN DEFAULT 1;

-- Create extraction_logs table for better monitoring
CREATE TABLE IF NOT EXISTS extraction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_name TEXT,
    extraction_method TEXT NOT NULL,
    status TEXT NOT NULL,
    records_extracted INTEGER DEFAULT 0,
    error_message TEXT,
    execution_time REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_service_plans_provider ON service_plans(provider_name);
CREATE INDEX IF NOT EXISTS idx_service_plans_extraction_method ON service_plans(extraction_method);
CREATE INDEX IF NOT EXISTS idx_providers_category ON providers(category);
CREATE INDEX IF NOT EXISTS idx_extraction_logs_timestamp ON extraction_logs(timestamp);

-- Update existing records with default values
UPDATE service_plans SET extraction_method = 'legacy_import' WHERE extraction_method IS NULL;
UPDATE providers SET extraction_method = 'manual_entry' WHERE extraction_method IS NULL;
UPDATE providers SET is_active = 1 WHERE is_active IS NULL;