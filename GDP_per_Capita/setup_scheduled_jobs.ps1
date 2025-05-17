# PowerShell script to create Windows Scheduled Tasks for GDP dashboard automation
# This will schedule both data fetch and analytics enhancement jobs to run daily when the laptop is on.

$projectDir = "C:\Users\shubhajoshi\GDP_per_Capita"
$pythonExe = "python"  # Use full path if not in PATH

# Data Fetch Task: runs daily at 2:00 AM if the laptop is on
$action1 = New-ScheduledTaskAction -Execute $pythonExe -Argument "data/fetch_data.py"
$trigger1 = New-ScheduledTaskTrigger -Daily -At 2:00am
$settings1 = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "GDP_Data_Fetch" -Action $action1 -Trigger $trigger1 -Settings $settings1 -WorkingDirectory $projectDir -Description "Fetches latest economic data for dashboard" -Force

# Analytics Enhancement Task: runs daily at 2:30 AM if the laptop is on
$action2 = New-ScheduledTaskAction -Execute $pythonExe -Argument "dashboard/enhance_analytics.py"
$trigger2 = New-ScheduledTaskTrigger -Daily -At 2:30am
$settings2 = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "GDP_Analytics_Enhance" -Action $action2 -Trigger $trigger2 -Settings $settings2 -WorkingDirectory $projectDir -Description "Retrains models and updates analytics for dashboard" -Force

Write-Host "Scheduled tasks created! You can view and manage them in Task Scheduler."
