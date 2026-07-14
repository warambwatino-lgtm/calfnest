from calfnest.consent import ConsentRecord, may_sync_personal_data, withdraw


def test_no_record_blocks_sync():
    assert may_sync_personal_data(None) is False


def test_ungranted_blocks_sync():
    rec = ConsentRecord("farmer-1", "welfare-alerts", "v1", "ussd", granted=False)
    assert may_sync_personal_data(rec) is False


def test_granted_allows_sync():
    rec = ConsentRecord("farmer-1", "welfare-alerts", "v1", "ussd", granted=True)
    assert may_sync_personal_data(rec) is True


def test_withdrawal_revokes_sync():
    rec = ConsentRecord("farmer-1", "welfare-alerts", "v1", "app", granted=True)
    withdraw(rec)
    assert rec.granted is False
    assert may_sync_personal_data(rec) is False
