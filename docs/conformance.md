# Binding the conformance suites

The suite classes intentionally do not begin with `Test`, so installing this package
does not collect them by itself. An adapter defines a `Test*` subclass and returns the
smallest fixture required by that suite. Inherited `test_*` methods are then collected
by pytest.

Use actual boundary evidence:

- `RuntimeObservation.observed_contexts` records real model inputs.
- `StartControl` and `ResponseControl` observe submission and acknowledgement at the
  real delivery boundary.
- effect and workspace fixtures inspect isolated files and network endpoints.
- persistence fixtures manipulate the adapter's actual backing storage.
- lifecycle controls are reserved for SDK-boundary lifecycle assertions; they must
  feed the same adapter ingestion path as native notifications.

Do not implement a universal fake harness to satisfy the suites. Do not compute
fixture expectations from `support`, `validate`, task state, events, or outcomes. A
passing test must compare the public port with evidence independently observed at the
boundary being claimed.

The suites are grouped by the facts an adapter can actually expose. Unsupported
features should be rejected honestly by the requirements matrix rather than simulated
as successful. Bind only suites whose fixture seam exists, while declaring every
`Capability` explicitly in each requirements profile, including `UnknownSupport`.

Important timing rules:

- suite timeouts are generous outer safety bounds, not advertised cleanup budgets;
- cleanup assertions compare elapsed time to `CleanupBudget.total` once per operation;
- cancelling a Python waiter or cleanup caller must not erase the underlying work or
  cleanup obligation;
- start and response acknowledgement loss must retain the submitted identity and
  prevent automatic duplicate delivery.

